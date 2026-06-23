import asyncio
import hashlib
import json
import re
from typing import TypedDict

import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient

from app.config import settings


class AgentState(TypedDict):
    topic: str
    niche: str
    sub_queries: list[str]
    raw_results: list[dict]
    clean_docs: list[dict]
    chunks: list[dict]


async def query_expansion_node(state: AgentState) -> AgentState:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = (
        f"Generate exactly 5 diverse sub-queries to research the topic: '{state['topic']}' "
        f"in the niche: '{state['niche']}'. "
        "Cover different angles: definitions, comparisons, use cases, limitations, recent developments. "
        "Return ONLY a JSON array of 5 strings, no explanation."
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    sub_queries: list[str] = json.loads(raw)
    return {**state, "sub_queries": sub_queries[:5]}


async def web_search_node(state: AgentState) -> AgentState:
    client = AsyncTavilyClient(api_key=settings.tavily_api_key)

    async def search_one(q: str) -> dict:
        try:
            result = await client.search(q, search_depth="basic", max_results=5)
            return {"query": q, "results": result.get("results", [])}
        except Exception:
            return {"query": q, "results": []}

    raw_results = await asyncio.gather(*[search_one(q) for q in state["sub_queries"]])
    return {**state, "raw_results": list(raw_results)}


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


async def content_extraction_node(state: AgentState) -> AgentState:
    seen_urls: set[str] = set()
    clean_docs: list[dict] = []

    for bucket in state["raw_results"]:
        for item in bucket.get("results", []):
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            text = _clean_text(item.get("content") or item.get("raw_content") or "")
            if _approx_tokens(text) < 200:
                continue

            clean_docs.append({
                "url": url,
                "title": item.get("title", ""),
                "text": text,
            })

    return {**state, "clean_docs": clean_docs}


async def chunk_embed_node(state: AgentState) -> AgentState:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    oai = AsyncOpenAI(api_key=settings.openai_api_key)

    all_chunks: list[dict] = []
    texts_to_embed: list[str] = []
    metadata_list: list[dict] = []

    for doc in state["clean_docs"]:
        for chunk_text in splitter.split_text(doc["text"]):
            texts_to_embed.append(chunk_text)
            metadata_list.append({"url": doc["url"], "title": doc["title"]})

    if not texts_to_embed:
        return {**state, "chunks": []}

    # Batch embed in groups of 100 (OpenAI limit is 2048 but keep batches small)
    embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i : i + batch_size]
        response = await oai.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        embeddings.extend([e.embedding for e in response.data])

    for text, embedding, metadata in zip(texts_to_embed, embeddings, metadata_list):
        all_chunks.append({"text": text, "embedding": embedding, "metadata": metadata})

    return {**state, "chunks": all_chunks}


def get_ingestion_agent():
    graph = StateGraph(AgentState)

    graph.add_node("query_expansion", query_expansion_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("content_extraction", content_extraction_node)
    graph.add_node("chunk_embed", chunk_embed_node)

    graph.set_entry_point("query_expansion")
    graph.add_edge("query_expansion", "web_search")
    graph.add_edge("web_search", "content_extraction")
    graph.add_edge("content_extraction", "chunk_embed")
    graph.add_edge("chunk_embed", END)

    return graph.compile()
