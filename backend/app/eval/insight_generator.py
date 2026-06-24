import json
import logging
import re

from app.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()


class InsightGenerator:
    def generate_query_insight(self, query: str, all_eval_scores: list[dict]) -> str:
        try:
            sorted_scores = sorted(
                all_eval_scores, key=lambda x: x["composite_score"], reverse=True
            )

            lines = []
            for s in sorted_scores:
                lines.append(
                    f"  {s['pipeline_id']}: composite={s['composite_score']:.3f} "
                    f"faithfulness={s.get('faithfulness', 0):.2f} "
                    f"relevancy={s.get('answer_relevancy', 0):.2f} "
                    f"correctness={s.get('judge_correctness', 0)} "
                    f"completeness={s.get('judge_completeness', 0)} "
                    f"groundedness={s.get('judge_groundedness', 0)} "
                    f"latency={s.get('latency_score', 0):.2f}"
                )
            formatted_scores = "\n".join(lines)

            prompt = (
                f"You are analyzing RAG pipeline benchmark results.\n\n"
                f"Query: {query}\n\n"
                f"Pipeline scores (sorted by composite score):\n{formatted_scores}\n\n"
                f"In 2-3 sentences explain:\n"
                f"1. Why the top pipeline won\n"
                f"2. Why the bottom pipeline struggled\n"
                f"3. The most interesting tradeoff in these results\n\n"
                f"Be specific and technical. Reference actual score differences."
            )

            return LLMClient().generate(prompt=prompt)
        except Exception as e:
            logger.error("InsightGenerator.generate_query_insight failed: %s", e)
            return "Insight generation failed."

    def generate_capability_profile(
        self, pipeline_id: str, dimension_scores: dict
    ) -> dict:
        _defaults = {
            "best_for": ["general question answering"],
            "avoid_when": ["specialized use cases"],
            "sweet_spot": "General purpose RAG queries.",
        }

        try:
            formatted_dimensions = "\n".join(
                f"  {k}: {v:.3f}" for k, v in dimension_scores.items()
            )

            prompt = (
                f"Given these RAG pipeline benchmark scores across query dimensions:\n"
                f"Pipeline: {pipeline_id}\n"
                f"{formatted_dimensions}\n\n"
                f"Return ONLY a JSON object:\n"
                f'{{\n'
                f'  "best_for": [list of 2-3 use cases this pipeline excels at],\n'
                f'  "avoid_when": [list of 2 scenarios to avoid this pipeline],\n'
                f'  "sweet_spot": "one sentence describing ideal use case"\n'
                f"}}"
            )

            raw = LLMClient().generate(prompt=prompt)
            data = json.loads(_strip_fences(raw))
            return {
                "best_for": data.get("best_for", _defaults["best_for"]),
                "avoid_when": data.get("avoid_when", _defaults["avoid_when"]),
                "sweet_spot": data.get("sweet_spot", _defaults["sweet_spot"]),
            }
        except json.JSONDecodeError:
            logger.error("InsightGenerator capability profile JSON parse failed for %s", pipeline_id)
            return dict(_defaults)
        except Exception as e:
            logger.error("InsightGenerator.generate_capability_profile failed: %s", e)
            return dict(_defaults)
