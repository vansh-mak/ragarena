import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.ingestion_agent import get_ingestion_agent
from app.config import settings
from app.database import get_db
from app.models.benchmark_run import BenchmarkRun
from app.pipelines.naive_rag import NaiveRAGPipeline
from app.pipelines.hyde_fusion import HyDEFusionPipeline
from app.pipelines.self_rag import SelfRAGPipeline
from app.pipelines.graph_rag import GraphRAGPipeline
from app.pipelines.agentic_rag import AgenticRAGPipeline
from app.pipelines.kag_cag import KAGCAGPipeline
from app.pipelines.vectorless_rag import VectorlessRAGPipeline
from app.tasks.orchestrator import run_benchmark_chord

router = APIRouter()

_ALL_PIPELINES = [
    NaiveRAGPipeline,
    HyDEFusionPipeline,
    SelfRAGPipeline,
    GraphRAGPipeline,
    AgenticRAGPipeline,
    KAGCAGPipeline,
    VectorlessRAGPipeline,
]


class IngestRequest(BaseModel):
    topic: str
    niche: str


class RunRequest(BaseModel):
    run_id: str
    queries: list[str]


@router.post("/ingest")
async def ingest(body: IngestRequest, db: AsyncSession = Depends(get_db)):
    run = BenchmarkRun(
        topic=body.topic,
        niche=body.niche,
        status="ingesting",
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    run_id = str(run.id)

    agent = get_ingestion_agent()
    state = await agent.ainvoke({
        "topic": body.topic,
        "niche": body.niche,
        "sub_queries": [],
        "raw_results": [],
        "clean_docs": [],
        "chunks": [],
    })

    chunks = state.get("chunks", [])

    for pipeline_cls in _ALL_PIPELINES:
        pipeline_cls(run_id=run_id).ingest(chunks)

    run.status = "ready"
    run.chunk_count = len(chunks)
    await db.commit()

    return {"run_id": run_id, "chunk_count": len(chunks)}


@router.post("/run")
async def run_benchmark(body: RunRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BenchmarkRun).where(BenchmarkRun.id == uuid.UUID(body.run_id))
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "ready":
        raise HTTPException(status_code=400, detail=f"Run is not ready (status={run.status})")

    # Set initial Redis progress
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    await r.set(f"run:{body.run_id}:status", "running")
    await r.set(f"run:{body.run_id}:progress", "0")
    await r.aclose()

    chord_result = run_benchmark_chord(body.run_id, body.queries)
    job_id = chord_result.id if hasattr(chord_result, "id") else str(chord_result)

    return {"run_id": body.run_id, "job_id": job_id}


@router.get("/{run_id}/status")
async def get_status(run_id: str):
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    status = await r.get(f"run:{run_id}:status")
    progress = await r.get(f"run:{run_id}:progress")
    await r.aclose()
    return {
        "run_id": run_id,
        "status": status or "unknown",
        "progress": progress or "0",
    }
