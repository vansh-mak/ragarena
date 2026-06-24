import asyncio
import logging
import uuid

import redis as redis_lib
from sqlalchemy import select, update

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.benchmark_run import BenchmarkRun
from app.models.eval_score import EvalScore
from app.models.pipeline_result import PipelineResult as PipelineResultModel
from app.eval.ragas_scorer import RagasScorer
from app.eval.llm_judge import LLMJudge
from app.eval.operational import OperationalScorer
from app.eval.aggregator import ScoreAggregator
from app.eval.insight_generator import InsightGenerator

logger = logging.getLogger(__name__)


async def _run_evaluation_async(run_id: str, queries: list[str]) -> None:
    async with AsyncSessionLocal() as session:
        stmt = select(PipelineResultModel).where(
            PipelineResultModel.run_id == uuid.UUID(run_id)
        )
        rows = (await session.execute(stmt)).scalars().all()

    # Build plain dicts and group by query_id
    query_groups: dict[str, list[dict]] = {}
    for row in rows:
        entry = {
            "pipeline_id": row.pipeline_id,
            "query_id": row.query_id,
            "query_text": row.query_text,
            "answer": row.answer or "",
            "context_chunks": row.context_chunks or [],
            "retrieval_ms": row.retrieval_ms or 0.0,
            "generation_ms": row.generation_ms or 0.0,
            "token_input": row.token_input or 0,
            "token_output": row.token_output or 0,
        }
        query_groups.setdefault(row.query_id, []).append(entry)

    ragas_scorer = RagasScorer()
    llm_judge = LLMJudge()
    op_scorer = OperationalScorer()
    aggregator = ScoreAggregator()
    insight_gen = InsightGenerator()

    async with AsyncSessionLocal() as session:
        for query_id, all_results in query_groups.items():
            query_text = all_results[0]["query_text"]
            all_eval_scores: list[dict] = []

            for pr in all_results:
                ragas = ragas_scorer.score(query_text, pr)
                judge = llm_judge.judge(query_text, pr["answer"], pr["context_chunks"])
                ops = op_scorer.score(pr, all_results)
                composite = aggregator.aggregate(ragas, judge, ops)

                session.add(EvalScore(
                    run_id=uuid.UUID(run_id),
                    pipeline_id=pr["pipeline_id"],
                    query_id=query_id,
                    faithfulness=ragas["faithfulness"],
                    answer_relevancy=ragas["answer_relevancy"],
                    context_precision=ragas["context_precision"],
                    context_recall=ragas["context_recall"],
                    judge_correctness=judge["correctness"],
                    judge_completeness=judge["completeness"],
                    judge_groundedness=judge["groundedness"],
                    latency_score=ops["latency_score"],
                    cost_usd=ops["cost_usd"],
                    composite_score=composite,
                ))

                all_eval_scores.append({
                    "pipeline_id": pr["pipeline_id"],
                    "composite_score": composite,
                    "faithfulness": ragas["faithfulness"],
                    "answer_relevancy": ragas["answer_relevancy"],
                    "judge_correctness": judge["correctness"],
                    "judge_completeness": judge["completeness"],
                    "judge_groundedness": judge["groundedness"],
                    "latency_score": ops["latency_score"],
                    "cost_usd": ops["cost_usd"],
                })

            await session.flush()

            insight = insight_gen.generate_query_insight(query_text, all_eval_scores)
            await session.execute(
                update(EvalScore)
                .where(EvalScore.run_id == uuid.UUID(run_id))
                .where(EvalScore.query_id == query_id)
                .values(insight=insight)
            )

        await session.execute(
            update(BenchmarkRun)
            .where(BenchmarkRun.id == uuid.UUID(run_id))
            .values(status="complete")
        )
        await session.commit()

    r = redis_lib.from_url(settings.redis_url, decode_responses=True)
    r.set(f"run:{run_id}:status", "complete")
    r.set(f"run:{run_id}:progress", "100")

    logger.info("Evaluation complete for run_id=%s", run_id)


async def run_evaluation(run_id: str, queries: list[str]) -> None:
    await _run_evaluation_async(run_id, queries)
