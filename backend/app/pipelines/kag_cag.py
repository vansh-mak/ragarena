import pickle
import time
import uuid
from pathlib import Path

import chromadb
import numpy as np

from app.config import settings
from app.llm_client import LLMClient

PIPELINE_ID = "kag_cag"
_SUMMARY_PROMPT = (
    "Summarise the key facts from these chunks as a structured knowledge summary. "
    "Be concise and factual.\nChunks:\n{batch_texts}"
)
_CACHE_THRESHOLD = 0.92


class KAGCAGPipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.collection = client.get_or_create_collection(
            name=f"kag_{run_id}",
            metadata={"hnsw:space": "cosine"},
        )
        self.cache_path = Path(settings.chromadb_path) / f"kag_cache_{run_id}"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.answer_cache: dict[str, dict] = {}

    def ingest(self, chunks: list[dict]) -> None:
        if not chunks:
            return

        # Step 1 — store raw chunks with precomputed embeddings
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{**(c.get("metadata") or {}), "type": "raw"} for c in chunks],
        )

        # Step 2 — knowledge augmentation: batch of 5, summarise, store back
        llm = LLMClient()
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = "\n\n".join(c["text"] for c in batch)
            summary = llm.generate(_SUMMARY_PROMPT.format(batch_texts=batch_texts))
            summary_emb = llm.embed([summary])[0]
            self.collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[summary_emb],
                documents=[summary],
                metadatas=[{"type": "knowledge_summary"}],
            )

        # Step 3 — persist (empty at ingest time, but creates the file)
        self._persist_cache()

    def _persist_cache(self) -> None:
        with open(self.cache_path / "cache.pkl", "wb") as f:
            pickle.dump(self.answer_cache, f)

    def _load_cache(self) -> None:
        cache_file = self.cache_path / "cache.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                self.answer_cache = pickle.load(f)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        llm = LLMClient()
        q_emb = np.array(llm.embed([query])[0])

        # Step 1 — cache check
        self._load_cache()
        for entry in self.answer_cache.values():
            cached_emb = np.array(entry["embedding"])
            similarity = float(np.dot(q_emb, cached_emb))
            if similarity >= _CACHE_THRESHOLD:
                return entry["chunks"]

        # Step 2 — cache miss: summaries first, then raw, merge and dedup
        n = max(1, self.collection.count())
        fetch_k = min(k, n)

        summary_results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=fetch_k,
            where={"type": "knowledge_summary"},
        )
        summary_docs = summary_results["documents"][0] if summary_results["documents"] else []

        raw_results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=fetch_k,
        )
        raw_docs = raw_results["documents"][0] if raw_results["documents"] else []

        # Summaries ranked first, then raw; deduplicate preserving order
        seen: set[str] = set()
        merged: list[str] = []
        for doc in summary_docs + raw_docs:
            if doc not in seen:
                seen.add(doc)
                merged.append(doc)
        chunks = merged[:k]

        # Step 3 — cache result
        cache_key = query
        self.answer_cache[cache_key] = {"embedding": q_emb.tolist(), "chunks": chunks}
        self._persist_cache()

        return chunks

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
