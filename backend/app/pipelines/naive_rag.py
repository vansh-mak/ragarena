import time
import uuid

import chromadb

from app.config import settings
from app.llm_client import LLMClient

PIPELINE_ID = "naive_rag"


class NaiveRAGPipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.collection = client.get_or_create_collection(
            name=f"naive_{run_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c.get("metadata", {}) for c in chunks],
        )

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        llm = LLMClient()
        query_embedding = llm.embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count() or 1),
        )
        return results["documents"][0] if results["documents"] else []

    def generate(self, query: str, query_id: str) -> dict:
        llm = LLMClient()

        t0 = time.perf_counter()
        context_chunks = self.retrieve(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        context_text = "\n\n".join(context_chunks)
        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        t1 = time.perf_counter()
        answer = llm.generate(prompt)
        generation_ms = (time.perf_counter() - t1) * 1000

        token_input = len(prompt.split())
        token_output = len(answer.split())

        return {
            "pipeline_id": PIPELINE_ID,
            "query_id": query_id,
            "answer": answer,
            "context_chunks": context_chunks,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "token_input": token_input,
            "token_output": token_output,
        }
