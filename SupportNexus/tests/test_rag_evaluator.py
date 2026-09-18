from evaluation.rag_evaluator import GoldCase, _percentile_nearest_rank, compute_metrics


def test_compute_metrics_uses_grade_gte_one_gold_and_top_five_ranks():
    cases = [
        GoldCase("one", "query one", frozenset({"a", "b"}), frozenset({"a"})),
        GoldCase("two", "query two", frozenset({"c"}), frozenset({"c"})),
    ]
    runs = [
        {"case_id": "one", "result_ids": ["x", "a", "z"], "latency_ms": 10},
        {"case_id": "two", "result_ids": ["c", "x"], "latency_ms": 100},
    ]

    metrics = compute_metrics(cases, runs)

    assert metrics == {
        "recall_at_5": 0.75,
        "mrr_at_5": 0.75,
        "hit_at_5": 1.0,
        "p95_latency_ms": 100.0,
        "mean_latency_ms": 55.0,
    }


def test_nearest_rank_p95_uses_the_95th_observation():
    assert _percentile_nearest_rank(list(range(1, 101)), 0.95) == 95
