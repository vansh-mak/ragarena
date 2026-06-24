import json
import pickle
import re
import time
import uuid
from pathlib import Path
from typing import TypedDict

import chromadb
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END

from app.config import settings
from app.llm_client import LLMClient

PIPELINE_ID = "agentic_rag"
_AGENT_PROMPT = (
    "You are a retrieval agent. Based on the query and chunks collected "
    "so far, decide what to do next.\n"
    "Query: {query}\n"
    "Chunks collected: {n_chunks}\n"
    "Iterations: {iteration}\n"
    "Available actions: vector_search, keyword_search, summarise, generate_answer\n"
    'Reply with JSON only: {{"action": "str", "reason": "str"}}'
)


class AgenticState(TypedDict):
    query: str
    query_id: str
    collected_chunks: list[str]
    tool_calls: list[str]
    answer: str
    iteration: int
    retrieval_ms: float
    generation_ms: float
    token_input: int
    token_output: int
    action: str  # tracks latest agent decision for routing


# ── tool functions ────────────────────────────────────────────────────────────

def vector_search(query: str, collection, k: int = 5) -> list[str]:
    llm = LLMClient()
    q_emb = llm.embed([query])[0]
    n = max(1, collection.count())
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(k, n),
    )
    return results["documents"][0] if results["documents"] else []


def keyword_search(query: str, bm25_path: Path, k: int = 5) -> list[str]:
    bm25_file = bm25_path / "bm25.pkl"
    chunks_file = bm25_path / "chunks.pkl"
    if not bm25_file.exists() or not chunks_file.exists():
        return []
    with open(bm25_file, "rb") as f:
        index: BM25Okapi = pickle.load(f)
    with open(chunks_file, "rb") as f:
        texts: list[str] = pickle.load(f)
    scores = index.get_scores(query.lower().split())
    import numpy as np
    top_idx = list(np.argsort(scores)[::-1][:k])
    return [texts[i] for i in top_idx]


def summarise_chunks(chunks: list[str], llm: LLMClient) -> str:
    joined = "\n\n".join(chunks)
    return llm.generate(
        f"Summarise the following passages into a concise paragraph:\n\n{joined}"
    )


# ── pipeline ──────────────────────────────────────────────────────────────────

class AgenticRAGPipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        chroma_client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.collection = chroma_client.get_or_create_collection(
            name=f"agentic_{run_id}",
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25_path = Path(settings.chromadb_path) / f"agentic_bm25_{run_id}"
        self.bm25_path.mkdir(parents=True, exist_ok=True)
        self._graph = self._build_graph()

    def ingest(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        # ChromaDB
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c.get("metadata", {}) for c in chunks],
        )
        # BM25
        texts = [c["text"] for c in chunks]
        index = BM25Okapi([t.lower().split() for t in texts])
        with open(self.bm25_path / "bm25.pkl", "wb") as f:
            pickle.dump(index, f)
        with open(self.bm25_path / "chunks.pkl", "wb") as f:
            pickle.dump(texts, f)

    def _build_graph(self):
        collection = self.collection
        bm25_path = self.bm25_path

        def _parse_action(raw: str) -> str:
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            try:
                return json.loads(raw).get("action", "generate_answer")
            except Exception:
                return "generate_answer"

        def agent_node(state: AgenticState) -> dict:
            llm = LLMClient()
            prompt = _AGENT_PROMPT.format(
                query=state["query"],
                n_chunks=len(state["collected_chunks"]),
                iteration=state["iteration"],
            )
            raw = llm.generate(prompt)
            action = _parse_action(raw)
            return {"action": action}

        def tool_executor_node(state: AgenticState) -> dict:
            llm = LLMClient()
            action = state["action"]
            new_chunks: list[str] = []

            t0 = time.perf_counter()
            if action == "vector_search":
                new_chunks = vector_search(state["query"], collection)
            elif action == "keyword_search":
                new_chunks = keyword_search(state["query"], bm25_path)
            elif action == "summarise" and state["collected_chunks"]:
                summary = summarise_chunks(state["collected_chunks"], llm)
                new_chunks = [summary]
            elapsed = (time.perf_counter() - t0) * 1000

            return {
                "collected_chunks": state["collected_chunks"] + new_chunks,
                "tool_calls": state["tool_calls"] + [action],
                "iteration": state["iteration"] + 1,
                "retrieval_ms": state["retrieval_ms"] + elapsed,
            }

        def generate_node(state: AgenticState) -> dict:
            llm = LLMClient()
            context = state["collected_chunks"]
            context_text = "\n\n".join(context) if context else "(no context retrieved)"
            prompt = (
                f"Answer the following question using only the provided context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {state['query']}\n\n"
                f"Answer:"
            )
            t0 = time.perf_counter()
            answer = llm.generate(prompt)
            gen_ms = (time.perf_counter() - t0) * 1000
            return {
                "answer": answer,
                "generation_ms": gen_ms,
                "token_input": len(prompt.split()),
                "token_output": len(answer.split()),
            }

        def route_after_tool(state: AgenticState) -> str:
            if state["action"] == "generate_answer" or state["iteration"] >= 4:
                return "generate"
            return "agent"

        graph = StateGraph(AgenticState)
        graph.add_node("agent", agent_node)
        graph.add_node("tool_executor", tool_executor_node)
        graph.add_node("generate", generate_node)

        graph.set_entry_point("agent")
        graph.add_edge("agent", "tool_executor")
        graph.add_conditional_edges("tool_executor", route_after_tool, {
            "generate": "generate",
            "agent": "agent",
        })
        graph.add_edge("generate", END)

        return graph.compile()

    def generate(self, query: str, query_id: str) -> dict:
        initial: AgenticState = {
            "query": query,
            "query_id": query_id,
            "collected_chunks": [],
            "tool_calls": [],
            "answer": "",
            "iteration": 0,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "token_input": 0,
            "token_output": 0,
            "action": "",
        }
        result = self._graph.invoke(initial)
        return {
            "pipeline_id": PIPELINE_ID,
            "query_id": query_id,
            "answer": result["answer"],
            "context_chunks": result["collected_chunks"],
            "retrieval_ms": result["retrieval_ms"],
            "generation_ms": result["generation_ms"],
            "token_input": result["token_input"],
            "token_output": result["token_output"],
        }
