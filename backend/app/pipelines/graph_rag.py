import json
import logging
import pickle
import re
import time
import uuid
from pathlib import Path

import chromadb
import networkx as nx
import numpy as np

from app.config import settings
from app.llm_client import LLMClient

logger = logging.getLogger(__name__)

PIPELINE_ID = "graph_rag"
_ENTITY_PROMPT = (
    "Extract named entities and relations from this text. "
    "Return JSON only, no explanation:\n"
    '{{"entities": [{{"name": "str", "type": "str"}}], '
    '"relations": [{{"src": "str", "rel": "str", "tgt": "str"}}]}}\n'
    "Text: {text}"
)
_QUERY_ENTITY_PROMPT = (
    "Extract named entities from the following query. "
    'Return JSON only: {{"entities": [{{"name": "str", "type": "str"}}]}}\n'
    "Query: {query}"
)


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    return re.sub(r"\s*```$", "", raw)


def _parse_entities(raw: str) -> dict:
    try:
        return json.loads(_strip_fences(raw))
    except Exception:
        return {"entities": [], "relations": []}


class GraphRAGPipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        client = chromadb.PersistentClient(path=settings.chromadb_path)
        self.collection = client.get_or_create_collection(
            name=f"graph_{run_id}",
            metadata={"hnsw:space": "cosine"},
        )
        self.graph_path = Path(settings.chromadb_path) / f"graph_{run_id}"
        self.graph_path.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()

    def ingest(self, chunks: list[dict]) -> None:
        if not chunks:
            return

        # Step 1 — store in ChromaDB with precomputed embeddings
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(
            ids=chunk_ids,
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c.get("metadata", {}) for c in chunks],
        )

        # Step 2 & 3 — extract entities/relations, build graph
        llm = LLMClient()
        for chunk_id, chunk in zip(chunk_ids, chunks):
            try:
                raw = llm.generate(_ENTITY_PROMPT.format(text=chunk["text"]))
                parsed = _parse_entities(raw)

                for ent in parsed.get("entities", []):
                    if not isinstance(ent, dict):
                        continue
                    name = ent.get("name", "").strip()
                    if not name:
                        continue
                    if name not in self.graph:
                        self.graph.add_node(name, type=ent.get("type", ""), chunk_ids=set())
                    self.graph.nodes[name]["chunk_ids"].add(chunk_id)

                for rel in parsed.get("relations", []):
                    if not isinstance(rel, dict):
                        continue
                    src = rel.get("src", "").strip()
                    tgt = rel.get("tgt", "").strip()
                    r = rel.get("rel", "").strip()
                    if src and tgt:
                        if src not in self.graph:
                            self.graph.add_node(src, type="", chunk_ids=set())
                        if tgt not in self.graph:
                            self.graph.add_node(tgt, type="", chunk_ids=set())
                        self.graph.add_edge(src, tgt, rel=r)
            except Exception:
                logger.warning("Entity extraction failed for chunk %s — skipping", chunk_id)

        # Step 4 — persist
        with open(self.graph_path / "graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        llm = LLMClient()

        # Step 1 — load graph
        graph_file = self.graph_path / "graph.pkl"
        if graph_file.exists():
            with open(graph_file, "rb") as f:
                graph: nx.DiGraph = pickle.load(f)
        else:
            graph = nx.DiGraph()

        # Step 2 — extract entities from query
        raw = llm.generate(_QUERY_ENTITY_PROMPT.format(query=query))
        parsed = _parse_entities(raw)
        query_entities = [
            e.get("name", "").strip()
            for e in parsed.get("entities", [])
            if isinstance(e, dict)
        ]

        # Build case-insensitive lookup
        node_lower = {n.lower(): n for n in graph.nodes}

        # Step 3 — 2-hop neighbourhood for matched entities
        matched_nodes: set[str] = set()
        for qe in query_entities:
            canonical = node_lower.get(qe.lower())
            if canonical:
                subgraph = nx.ego_graph(graph, canonical, radius=2)
                matched_nodes.update(subgraph.nodes)

        # Step 4 — collect chunk IDs from all matched nodes
        chunk_ids: list[str] = []
        for node in matched_nodes:
            chunk_ids.extend(graph.nodes[node].get("chunk_ids", set()))
        chunk_ids = list(set(chunk_ids))

        # Step 5 — fall back to ChromaDB cosine if no graph match
        if not chunk_ids:
            q_emb = llm.embed([query])[0]
            n = max(1, self.collection.count())
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=min(k, n),
            )
            return results["documents"][0] if results["documents"] else []

        # Step 6 — fetch chunks from ChromaDB, rerank by cosine similarity
        fetched = self.collection.get(
            ids=chunk_ids,
            include=["embeddings", "documents"],
        )
        docs = fetched.get("documents")
        if docs is None:
            docs = []
        embeddings = fetched.get("embeddings")
        if embeddings is None:
            embeddings = []

        if not docs:
            return []

        q_emb = np.array(llm.embed([query])[0])
        scores = [float(np.dot(q_emb, np.array(emb))) for emb in embeddings]
        ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
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
