import asyncio
import uuid

import redis as redis_lib
from celery import chord, group

from app.config import settings
from app.tasks.celery_app import celery_app
from app.tasks.pipeline_tasks import (
    run_naive_rag,
    run_hyde_fusion,
    run_self_rag,
    run_graph_rag,
    run_agentic_rag,
    run_kag_cag,
    run_vectorless,
)


async def _persist_results(results: list[dict], run_id: str) -> None:
    from app.database import AsyncSessionLocal
    from app.models.pipeline_result import PipelineResult

    async with AsyncSessionLocal() as session:
        for r in results:
            if not r:
                continue
            record = PipelineResult(
                run_id=uuid.UUID(run_id),
                pipeline_id=r["pipeline_id"],
                query_id=r["query_id"],
                query_text=r.get("query_text", ""),
                answer=r.get("answer"),
                context_chunks=r.get("context_chunks"),
                retrieval_ms=r.get("retrieval_ms"),
                generation_ms=r.get("generation_ms"),
                token_input=r.get("token_input"),
                token_output=r.get("token_output"),
            )
            session.add(record)
        await session.commit()


@celery_app.task
def aggregate_results(results: list[dict], run_id: str, queries: list[str]) -> None:
    asyncio.run(_persist_results(results, run_id))

    r = redis_lib.from_url(settings.redis_url, decode_responses=True)
    r.set(f"run:{run_id}:status", "eval_ready")
    r.set(f"run:{run_id}:progress", "100")

    from app.tasks.eval_tasks import run_eval_task
    run_eval_task.apply_async(args=[run_id, queries], queue="eval")


def run_benchmark_chord(run_id: str, queries: list[str]):
    _PIPELINE_TASKS = [
        run_naive_rag,
        run_hyde_fusion,
        run_self_rag,
        run_graph_rag,
        run_agentic_rag,
        run_kag_cag,
        run_vectorless,
    ]

    all_sigs = []
    for i, query in enumerate(queries):
        query_id = f"q{i:03d}"
        for task_fn in _PIPELINE_TASKS:
            all_sigs.append(task_fn.s(run_id, query, query_id))

    callback = aggregate_results.s(run_id, queries)
    result = chord(group(all_sigs))(callback)
    return result
