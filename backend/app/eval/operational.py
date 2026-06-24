class OperationalScorer:
    def score(self, pipeline_result: dict, all_results: list[dict]) -> dict:
        retrieval_ms = int(pipeline_result["retrieval_ms"])
        generation_ms = int(pipeline_result["generation_ms"])
        total_ms = retrieval_ms + generation_ms

        all_totals = [
            int(r["retrieval_ms"]) + int(r["generation_ms"]) for r in all_results
        ]
        max_ms = max(all_totals)
        min_ms = min(all_totals)

        if max_ms == min_ms:
            latency_score = 1.0
        else:
            latency_score = 1.0 - (total_ms - min_ms) / (max_ms - min_ms)

        return {
            "total_ms": total_ms,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "latency_score": round(latency_score, 4),
            "cost_usd": 0.0,
            "cost_score": 1.0,
            "token_input": int(pipeline_result["token_input"]),
            "token_output": int(pipeline_result["token_output"]),
            "chunks_fetched": len(pipeline_result["context_chunks"]),
        }
