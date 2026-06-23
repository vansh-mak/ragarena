import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.ingestion_agent import get_ingestion_agent
from app.database import get_db
from app.models.benchmark_run import BenchmarkRun
from app.models.pipeline_result import PipelineResult
from app.pipelines.naive_rag import NaiveRAGPipeline

router = APIRouter()


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

    pipeline = NaiveRAGPipeline(run_id=run_id)
    pipeline.ingest(chunks)

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

    pipeline = NaiveRAGPipeline(run_id=body.run_id)
    results = []

    for i, query_text in enumerate(body.queries):
        query_id = f"q{i:03d}"
        pipeline_result = pipeline.generate(query=query_text, query_id=query_id)

        record = PipelineResult(
            run_id=run.id,
            pipeline_id=pipeline_result["pipeline_id"],
            query_id=query_id,
            query_text=query_text,
            answer=pipeline_result["answer"],
            context_chunks=pipeline_result["context_chunks"],
            retrieval_ms=pipeline_result["retrieval_ms"],
            generation_ms=pipeline_result["generation_ms"],
            token_input=pipeline_result["token_input"],
            token_output=pipeline_result["token_output"],
        )
        db.add(record)
        results.append(pipeline_result)

    await db.commit()
    return {"run_id": body.run_id, "results": results}
