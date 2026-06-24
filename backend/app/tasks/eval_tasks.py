import asyncio

from app.tasks.celery_app import celery_app
from app.eval.eval_engine import run_evaluation


@celery_app.task(bind=True, max_retries=1, queue="eval")
def run_eval_task(self, run_id: str, queries: list[str]) -> None:
    try:
        asyncio.run(run_evaluation(run_id, queries))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)
