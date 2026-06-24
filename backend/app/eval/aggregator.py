from dataclasses import dataclass, field
from statistics import mean


@dataclass
class ScoringConfig:
    weights: dict = field(default_factory=lambda: {"ragas": 0.5, "judge": 0.3, "ops": 0.2})

    def validate(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


class ScoreAggregator:
    def aggregate(
        self,
        ragas_scores: dict,
        judge_scores: dict,
        operational_scores: dict,
        weights: dict = None,
    ) -> float:
        if weights is None:
            weights = {"ragas": 0.5, "judge": 0.3, "ops": 0.2}

        ragas_avg = mean([
            ragas_scores["faithfulness"],
            ragas_scores["answer_relevancy"],
            ragas_scores["context_precision"],
            ragas_scores["context_recall"],
        ])

        judge_raw = mean([
            judge_scores["correctness"],
            judge_scores["completeness"],
            judge_scores["groundedness"],
        ])
        judge_avg = (judge_raw - 1) / 4

        ops_avg = mean([
            operational_scores["latency_score"],
            operational_scores["cost_score"],
        ])

        composite = (
            ragas_avg * weights["ragas"]
            + judge_avg * weights["judge"]
            + ops_avg * weights["ops"]
        )
        return round(composite, 4)
