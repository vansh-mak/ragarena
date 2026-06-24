import pickle
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.llm_client import LLMClient

PIPELINE_ID = "vectorless"
_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(_RERANKER_MODEL_NAME)
    return _reranker


class VectorlessRAGPipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.index_path = Path(settings.chromadb_path) / f"vectorless_{run_id}"
        self.index_path.mkdir(parents=True, exist_ok=True)

    def ingest(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        tokenized = [t.lower().split() for t in texts]
        index = BM25Okapi(tokenized)
        with open(self.index_path / "bm25.pkl", "wb") as f:
            pickle.dump(index, f)
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(texts, f)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        with open(self.index_path / "bm25.pkl", "rb") as f:
            index: BM25Okapi = pickle.load(f)
        with open(self.index_path / "chunks.pkl", "rb") as f:
            texts: list[str] = pickle.load(f)

        tokenized_query = query.lower().split()
        scores = index.get_scores(tokenized_query)

        top_2k_idx = np.argsort(scores)[::-1][: k * 2]
        candidates = [texts[i] for i in top_2k_idx]

        reranker = get_reranker()
        pairs = [(query, doc) for doc in candidates]
        rerank_scores = reranker.predict(pairs)

        ranked = sorted(zip(rerank_scores, candidates), key=lambda x: -x[0])
        return [doc for _, doc in ranked[:k]]

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
