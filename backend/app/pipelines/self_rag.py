import time
import uuid
from typing import TypedDict

import chromadb
from langgraph.graph import StateGraph, END

from app.config import settings
from app.llm_client import LLMClient

PIPELINE_ID = "self_rag"


class SelfRAGState(TypedDict):
    query: str
    query_id: str
    chunks: list[str]
    passed_chunks: list[str]
    answer: str
    retrieval_ms: float
    generation_ms: float
    token_input: int
    token_output: int
    attempt: int


class SelfRAGPipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.collection = client.get_or_create_collection(
            name=f"selfrag_{run_id}",
            metadata={"hnsw:space": "cosine"},
        )
        self._graph = self._build_graph()

    def ingest(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c.get("metadata", {}) for c in chunks],
        )

    def _build_graph(self):
        collection = self.collection

        def retrieve_node(state: SelfRAGState) -> dict:
            llm = LLMClient()
            t0 = time.perf_counter()
            query_emb = llm.embed([state["query"]])[0]
            n = max(1, collection.count())
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=min(5, n),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            docs = results["documents"][0] if results["documents"] else []
            return {
                "chunks": docs,
                "attempt": state["attempt"] + 1,
                "retrieval_ms": state["retrieval_ms"] + elapsed,
            }

        def grade_node(state: SelfRAGState) -> dict:
            llm = LLMClient()
            passed = []
            for chunk in state["chunks"]:
                resp = llm.generate(
                    f"Is this chunk relevant to answer the question?\n"
                    f"Question: {state['query']}\n"
                    f"Chunk: {chunk}\n"
                    f"Reply with only yes or no."
                )
                if resp.strip().lower().startswith("yes"):
                    passed.append(chunk)
            return {"passed_chunks": passed}

        def decide(state: SelfRAGState) -> str:
            if len(state["passed_chunks"]) >= 2:
                return "generate"
            if state["attempt"] < 2:
                return "requery"
            return "generate"

        def requery_node(state: SelfRAGState) -> dict:
            llm = LLMClient()
            new_query = llm.generate(
                f"The following search query did not return enough relevant results. "
                f"Reformulate it to be more specific and likely to find relevant information.\n"
                f"Original query: {state['query']}\n"
                f"Return only the reformulated query, no explanation."
            )
            return {"query": new_query.strip()}

        def generate_node(state: SelfRAGState) -> dict:
            llm = LLMClient()
            context = state["passed_chunks"] or state["chunks"]
            context_text = "\n\n".join(context)
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

        graph = StateGraph(SelfRAGState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("grade", grade_node)
        graph.add_node("requery", requery_node)
        graph.add_node("generate", generate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", decide, {
            "generate": "generate",
            "requery": "requery",
        })
        graph.add_edge("requery", "retrieve")
        graph.add_edge("generate", END)

        return graph.compile()

    def generate(self, query: str, query_id: str) -> dict:
        initial: SelfRAGState = {
            "query": query,
            "query_id": query_id,
            "chunks": [],
            "passed_chunks": [],
            "answer": "",
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "token_input": 0,
            "token_output": 0,
            "attempt": 0,
        }
        result = self._graph.invoke(initial)
        context = result["passed_chunks"] or result["chunks"]
        return {
            "pipeline_id": PIPELINE_ID,
            "query_id": query_id,
            "answer": result["answer"],
            "context_chunks": context,
            "retrieval_ms": result["retrieval_ms"],
            "generation_ms": result["generation_ms"],
            "token_input": result["token_input"],
            "token_output": result["token_output"],
        }
