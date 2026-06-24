import json
import logging
import re

from app.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert evaluator for RAG (Retrieval Augmented Generation) systems. "
    "You evaluate answers strictly and objectively."
)

_PROMPT_TEMPLATE = """Evaluate this RAG system answer on 3 dimensions.

Question: {query}

Retrieved Context:
{context_text}

Answer: {answer}

Score each dimension from 1-5:
- correctness: Is every factual claim in the answer accurate?
- completeness: Does the answer cover all aspects of the question?
- groundedness: Is every claim traceable to the retrieved context?

Reply with ONLY a JSON object, no explanation:
{{"correctness": X, "completeness": X, "groundedness": X, "justification": "one sentence"}}"""

_DEFAULTS = {"correctness": 3, "completeness": 3, "groundedness": 3, "justification": "parse error"}


def _clamp_score(value) -> int:
    try:
        v = int(value)
        return v if 1 <= v <= 5 else 3
    except (TypeError, ValueError):
        return 3


class LLMJudge:
    def judge(self, query: str, answer: str, context_chunks: list[str]) -> dict:
        context_text = "\n\n".join(context_chunks) if context_chunks else "(no context)"
        prompt = _PROMPT_TEMPLATE.format(
            query=query,
            context_text=context_text,
            answer=answer,
        )

        try:
            raw = LLMClient().generate(prompt=prompt, system=_SYSTEM)
            # Strip markdown fences
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            data = json.loads(cleaned)
            return {
                "correctness": _clamp_score(data.get("correctness")),
                "completeness": _clamp_score(data.get("completeness")),
                "groundedness": _clamp_score(data.get("groundedness")),
                "justification": str(data.get("justification", "")).strip() or "no justification",
            }
        except json.JSONDecodeError:
            logger.error("LLMJudge JSON parse failed for query=%r", query)
            return dict(_DEFAULTS)
        except Exception as e:
            logger.error("LLMJudge failed for query=%r: %s", query, e)
            return dict(_DEFAULTS)
