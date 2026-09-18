"""Reproducible retrieval evaluation for the SupportNexus RAG baseline and full pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import httpx


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    query: str
    relevant_ids: frozenset[str]
    direct_ids: frozenset[str]


def _load_gold_cases(path: Path) -> tuple[Dict[str, Any], List[GoldCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        GoldCase(
            case_id=str(item["id"]),
            query=str(item["query"]),
            relevant_ids=frozenset(str(value) for value in item["relevant_chunk_ids"]),
            direct_ids=frozenset(str(value) for value in item["direct_chunk_ids"]),
        )
        for item in payload["cases"]
    ]
    if not cases:
        raise ValueError("gold set has no cases")
    if any(not case.relevant_ids for case in cases):
        raise ValueError("every gold case needs at least one grade >= 1 chunk")
    if any(not case.direct_ids for case in cases):
        raise ValueError("every gold case needs at least one grade == 2 chunk")
    return payload, cases


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _first_relevant_rank(result_ids: Iterable[str], relevant_ids: frozenset[str]) -> int | None:
    for rank, result_id in enumerate(result_ids, start=1):
        if result_id in relevant_ids:
            return rank
    return None


def compute_metrics(cases: Sequence[GoldCase], runs: Sequence[Dict[str, Any]], top_k: int = 5) -> Dict[str, float]:
    by_case = {str(run["case_id"]): run for run in runs}
    if set(by_case) != {case.case_id for case in cases}:
        raise ValueError("run results do not match the gold cases")

    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    hit_count = 0
    latency_values: List[float] = []

    for case in cases:
        run = by_case[case.case_id]
        result_ids = [str(value) for value in run["result_ids"][:top_k]]
        matched = set(result_ids).intersection(case.relevant_ids)
        recall_sum += len(matched) / len(case.relevant_ids)
        rank = _first_relevant_rank(result_ids, case.relevant_ids)
        if rank is not None:
            hit_count += 1
            reciprocal_rank_sum += 1.0 / rank
        latency_values.append(float(run["latency_ms"]))

    total = len(cases)
    return {
        "recall_at_5": round(recall_sum / total, 6),
        "mrr_at_5": round(reciprocal_rank_sum / total, 6),
        "hit_at_5": round(hit_count / total, 6),
        "p95_latency_ms": round(_percentile_nearest_rank(latency_values, 0.95), 3),
        "mean_latency_ms": round(sum(latency_values) / total, 3),
    }


def _verify_index(client: httpx.Client, base_url: str, gold: Dict[str, Any]) -> Dict[str, Any]:
    response = client.get(f"{base_url}/knowledge/chunks", params={"limit": 500})
    response.raise_for_status()
    payload = response.json()
    chunks = payload.get("chunks", [])
    expected_total = int(gold["expected_total_chunks"])
    if int(payload.get("total_chunks", -1)) != expected_total:
        raise RuntimeError(
            f"index chunk count mismatch: expected {expected_total}, got {payload.get('total_chunks')}"
        )
    index_ids = {str(chunk["id"]) for chunk in chunks}
    missing = sorted(set(gold["judged_chunk_ids"]) - index_ids)
    if missing:
        raise RuntimeError(f"index is missing {len(missing)} judged chunks, e.g. {missing[:3]}")
    fingerprint = hashlib.sha256("\n".join(sorted(index_ids)).encode("utf-8")).hexdigest()
    return {
        "total_chunks": int(payload["total_chunks"]),
        "judged_chunks_verified": len(gold["judged_chunk_ids"]),
        "index_id_sha256": fingerprint,
    }


def _warm_up(client: httpx.Client, base_url: str, endpoint: str, *, full_pipeline: bool) -> None:
    params: Dict[str, Any] = {"query": "RAG 检索服务预热，不计入评测结果", "top_k": 5}
    if full_pipeline:
        params["use_cache"] = "false"
    response = client.post(f"{base_url}{endpoint}", params=params)
    response.raise_for_status()
    if full_pipeline and response.json().get("reranked") is not True:
        raise RuntimeError("full pipeline warm-up did not apply Qwen3-Rerank")


def _run_profile(
    client: httpx.Client,
    base_url: str,
    endpoint: str,
    cases: Sequence[GoldCase],
    *,
    profile_name: str,
) -> Dict[str, Any]:
    is_full = profile_name == "full"
    _warm_up(client, base_url, endpoint, full_pipeline=is_full)
    runs: List[Dict[str, Any]] = []

    for case in cases:
        params: Dict[str, Any] = {"query": case.query, "top_k": 5}
        if is_full:
            params["use_cache"] = "false"
        started = time.perf_counter()
        response = client.post(f"{base_url}{endpoint}", params=params)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.json()

        if not payload.get("success", True):
            raise RuntimeError(f"{profile_name} failed for {case.case_id}: {payload.get('error')}")
        if is_full:
            if payload.get("reranked") is not True:
                raise RuntimeError(f"Qwen3-Rerank was not applied for {case.case_id}")
            if payload.get("cached") is True:
                raise RuntimeError(f"cache was unexpectedly hit for {case.case_id}")
        elif payload.get("reranked") is not False:
            raise RuntimeError(f"baseline unexpectedly reports reranked for {case.case_id}")

        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 5:
            raise RuntimeError(f"{profile_name} returned {len(results) if isinstance(results, list) else 'invalid'} results for {case.case_id}")
        runs.append({
            "case_id": case.case_id,
            "query": case.query,
            "result_ids": [str(item.get("id", "")) for item in results],
            "result_titles": [str(item.get("title", "")) for item in results],
            "latency_ms": round(latency_ms, 3),
        })

    return {
        "endpoint": endpoint,
        "cache_enabled": False,
        "metrics": compute_metrics(cases, runs),
        "runs": runs,
    }


def run_evaluation(gold_path: Path, base_url: str, timeout_s: float) -> Dict[str, Any]:
    gold, cases = _load_gold_cases(gold_path)
    base_url = base_url.rstrip("/")
    with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        index_snapshot = _verify_index(client, base_url, gold)
        baseline = _run_profile(client, base_url, "/search/baseline", cases, profile_name="baseline")
        full = _run_profile(client, base_url, "/search", cases, profile_name="full")

    return {
        "schema_version": "rag-evaluation-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_file": str(gold_path),
        "gold_file_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "relevance_policy": gold["relevance_policy"]["primary"],
        "case_count": len(cases),
        "top_k": 5,
        "latency_definition": "Sequential HTTP client wall-clock time, cache disabled, after one unscored warm-up request per profile.",
        "index_snapshot": index_snapshot,
        "profiles": {"baseline": baseline, "full": full},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SupportNexus baseline and full RAG retrieval.")
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_evaluation(args.gold_file, args.base_url, args.timeout_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "case_count": report["case_count"],
        "baseline": report["profiles"]["baseline"]["metrics"],
        "full": report["profiles"]["full"]["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
