from app.tasks.celery_app import celery_app


def _task(pipeline_cls, run_id: str, query: str, query_id: str) -> dict:
    pipeline = pipeline_cls(run_id=run_id)
    return pipeline.generate(query=query, query_id=query_id)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def run_naive_rag(self, run_id: str, query: str, query_id: str) -> dict:
    try:
        from app.pipelines.naive_rag import NaiveRAGPipeline
        return _task(NaiveRAGPipeline, run_id, query, query_id)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def run_hyde_fusion(self, run_id: str, query: str, query_id: str) -> dict:
    try:
        from app.pipelines.hyde_fusion import HyDEFusionPipeline
        return _task(HyDEFusionPipeline, run_id, query, query_id)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def run_self_rag(self, run_id: str, query: str, query_id: str) -> dict:
    try:
        from app.pipelines.self_rag import SelfRAGPipeline
        return _task(SelfRAGPipeline, run_id, query, query_id)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def run_graph_rag(self, run_id: str, query: str, query_id: str) -> dict:
    try:
        from app.pipelines.graph_rag import GraphRAGPipeline
        return _task(GraphRAGPipeline, run_id, query, query_id)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def run_agentic_rag(self, run_id: str, query: str, query_id: str) -> dict:
    try:
        from app.pipelines.agentic_rag import AgenticRAGPipeline
        return _task(AgenticRAGPipeline, run_id, query, query_id)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def run_kag_cag(self, run_id: str, query: str, query_id: str) -> dict:
    try:
        from app.pipelines.kag_cag import KAGCAGPipeline
        return _task(KAGCAGPipeline, run_id, query, query_id)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def run_vectorless(self, run_id: str, query: str, query_id: str) -> dict:
    try:
        from app.pipelines.vectorless_rag import VectorlessRAGPipeline
        return _task(VectorlessRAGPipeline, run_id, query, query_id)
    except Exception as exc:
        raise self.retry(exc=exc)
