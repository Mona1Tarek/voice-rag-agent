
import math
import statistics


def summarize(rows: list[dict]) -> dict:
    def stats(key):
        values = []

        for row in rows:
            value = row.get(key)

            # Timing/token metrics are stored at the top level.
            # Quality diagnostics are stored inside row["diagnostics"].
            if value is None:
                value = (row.get("diagnostics") or {}).get(key)

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)
                if math.isfinite(value):
                    values.append(value)

        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "p95": None,
            }

        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            # Nearest-rank percentile: the 95th percentile is
            # the smallest value covering at least 95% of observations.
            "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
        }

    n = len(rows)
    successful = sum(not row.get("error") for row in rows)

    abstention_matches = []
    for row in rows:
        value = row.get("abstention_heuristic_match")
        if value is None:
            value = (row.get("diagnostics") or {}).get(
                "abstention_heuristic_match"
            )
        if isinstance(value, bool):
            abstention_matches.append(value)

    return {
        "num_questions": n,
        "successful_requests": successful,
        "request_success_rate": successful / n if n else None,
        "error_count": n - successful,
        **{
            key: stats(key)
            for key in (
                "end_to_end_latency_ms",
                "retrieval_latency_ms",
                "llm_latency_ms",
                "ttft_ms",
                "generation_duration_ms",
                "output_tokens_per_second",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "citation_precision",
                "citation_recall",
                "citation_f1",
                "citation_id_validity",
            )
        },
        "abstention_heuristic_match_rate": (
            sum(abstention_matches) / len(abstention_matches)
            if abstention_matches
            else None
        ),
        "quality_note": (
            "Correctness, faithfulness, semantic citation support, and "
            "language quality require human review or a separately "
            "validated judge."
        ),
    }
