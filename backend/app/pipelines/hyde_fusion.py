import json
import re
import time
import uuid
from collections import Counter

import chromadb

from app.config import settings
from app.llm_client import LLMClient

PIPELINE_ID = "hyde_fusion"


class HyDEFusionPipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.collection = client.get_or_create_collection(
            name=f"hyde_{run_id}",
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
        n_docs = max(1, self.collection.count())

        # Step 1 — HyDE: generate hypothetical answer
        hyde_answer = llm.generate(
            f"Write a short factual answer to this question as if you were an expert: {query}"
        )

        # Step 2 — Query variants: 3 alternative phrasings
        raw = llm.generate(
            f"Produce 3 alternative phrasings of this question as a JSON array of strings, "
            f"no explanation: {query}"
        )
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        try:
            variants: list[str] = json.loads(raw)[:3]
        except Exception:
            variants = []

        # Step 3 — Embed all 5 texts
        texts_to_embed = [query, hyde_answer] + variants
        embeddings = llm.embed(texts_to_embed)

        # Step 4 — Retrieve top-k from ChromaDB for each embedding
        fetch_k = min(k * 2, n_docs)
        all_retrieved: list[str] = []
        for emb in embeddings:
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=fetch_k,
            )
            all_retrieved.extend(results["documents"][0] if results["documents"] else [])

        # Step 5 — Deduplicate, rerank by frequency
        freq = Counter(all_retrieved)
        unique_chunks = sorted(freq.keys(), key=lambda t: -freq[t])
        return unique_chunks[:k]

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

        return {
            "pipeline_id": PIPELINE_ID,
            "query_id": query_id,
            "answer": answer,
            "context_chunks": context_chunks,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "token_input": len(prompt.split()),
            "token_output": len(answer.split()),
        }
