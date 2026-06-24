import logging
import sys
import types

from app.config import settings

logger = logging.getLogger(__name__)

# ragas 0.4.x imports ChatVertexAI from langchain_community.chat_models.vertexai
# which was removed in langchain-community 0.4.x — inject a stub before ragas loads.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAI:
        pass

    _stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.metrics._answer_relevance import AnswerRelevancy  # noqa: E402
from ragas.metrics._context_precision import ContextPrecision  # noqa: E402
from ragas.metrics._context_recall import ContextRecall  # noqa: E402
from ragas.metrics._faithfulness import Faithfulness  # noqa: E402

from langchain_ollama import OllamaLLM as Ollama  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402


def _safe_float(value) -> float:
    try:
        v = float(value)
        if v != v:  # NaN
            return 0.0
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.0


class RagasScorer:
    def score(self, query: str, pipeline_result: dict) -> dict:
        defaults = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

        try:
            context_chunks = pipeline_result.get("context_chunks") or []
            answer = pipeline_result.get("answer", "")

            sample = SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=context_chunks,
                reference="",
            )
            dataset = EvaluationDataset(samples=[sample])

            llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
            embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embed_model,
            )

            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ]

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
                show_progress=False,
            )

            scores = result.scores[0] if result.scores else {}
            return {
                "faithfulness": _safe_float(scores.get("faithfulness")),
                "answer_relevancy": _safe_float(scores.get("answer_relevancy")),
                "context_precision": _safe_float(scores.get("context_precision")),
                "context_recall": _safe_float(scores.get("context_recall")),
            }

        except Exception as e:
            logger.error("RagasScorer failed for query=%r: %s", query, e)
            return defaults
