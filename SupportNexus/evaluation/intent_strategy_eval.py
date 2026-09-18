"""Compare fused intent recognition against single-path strategies.

Usage:
    python evaluation/intent_strategy_eval.py
    python evaluation/intent_strategy_eval.py --strategies fusion,llm,embedding,pattern
    python evaluation/intent_strategy_eval.py --output data/eval/intent_strategy_report.json

The script evaluates single-intent recognition quality and efficiency:
    1. fine intent accuracy and macro-F1
    2. intent group accuracy and macro-F1
    3. average/P95 latency, error rate, and optional token-based cost
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
import types
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    env_path = ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)

try:
    import anthropic  # noqa: F401
except ModuleNotFoundError:
    anthropic_stub = types.ModuleType("anthropic")

    class _OfflineMessages:
        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError(
                "anthropic is not installed; install requirements or run only "
                "embedding/pattern for local offline evaluation"
            )

    class AsyncAnthropic:  # type: ignore[no-redef]
        """Tiny offline stub so local embedding/pattern paths can be evaluated."""

        def __init__(self, **kwargs: Any) -> None:
            self.messages = _OfflineMessages()

    anthropic_stub.AsyncAnthropic = AsyncAnthropic
    sys.modules["anthropic"] = anthropic_stub

from core.intent_recognizer import IntentCategory, IntentRecognizer


DEFAULT_CASES_PATH = ROOT / "data" / "eval" / "intent_strategy_cases.json"
DEFAULT_REPORT_PATH = ROOT / "data" / "eval" / "intent_strategy_report.json"
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")


@dataclass
class CaseResult:
    id: str
    message: str
    strategy: str
    expected_intent: str
    predicted_intent: str
    expected_intent_group: str
    predicted_intent_group: str
    intent_ok: bool
    group_ok: bool
    latency_ms: float
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_input_tokens: int = 0
    confidence: float = 0.0
    difficulty: str = ""
    tags: List[str] = field(default_factory=list)
    probe: str = ""
    error: Optional[str] = None


async def _predict(
    recognizer: IntentRecognizer,
    message: str,
    strategy: str,
) -> tuple[IntentCategory, str, float, Optional[str]]:
    try:
        if strategy == "fusion":
            result = await recognizer.recognize(message)
            return result.intent, result.intent_group, result.confidence, None
        if strategy == "llm":
            raw = await recognizer._llm_recognize(message, history=None)
            error = str(raw.get("error") or "") if raw.get("failed") else None
            intent = raw.get("intent", IntentCategory.OTHER)
            group = raw.get("intent_group") or recognizer._intent_group(intent)
            return intent, group, float(raw.get("confidence", 0.0) or 0.0), error
        if strategy == "embedding":
            raw = await recognizer._embedding_recognize(message)
            intent = raw.get("intent", IntentCategory.OTHER)
            return intent, recognizer._intent_group(intent), float(raw.get("confidence", 0.0) or 0.0), None
        if strategy == "pattern":
            raw = recognizer._pattern_recognize(message)
            intent = raw.get("intent", IntentCategory.OTHER)
            return intent, recognizer._intent_group(intent), float(raw.get("confidence", 0.0) or 0.0), None
        raise ValueError(f"unknown strategy: {strategy}")
    except Exception as ex:
        return IntentCategory.OTHER, "other", 0.0, f"{type(ex).__name__}: {ex}"


async def evaluate_strategy(
    recognizer: IntentRecognizer,
    cases: List[Dict[str, Any]],
    strategy: str,
) -> List[CaseResult]:
    results: List[CaseResult] = []
    for case in cases:
        usage_before = recognizer.usage_snapshot()
        t0 = time.monotonic()
        predicted_intent, predicted_group, confidence, error = await _predict(
            recognizer,
            case["message"],
            strategy,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        usage_after = recognizer.usage_snapshot()

        results.append(CaseResult(
            id=case["id"],
            message=case["message"],
            strategy=strategy,
            expected_intent=case["expected_intent"],
            predicted_intent=predicted_intent.value,
            expected_intent_group=case["expected_intent_group"],
            predicted_intent_group=predicted_group,
            intent_ok=predicted_intent.value == case["expected_intent"],
            group_ok=predicted_group == case["expected_intent_group"],
            latency_ms=round(latency_ms, 1),
            llm_input_tokens=usage_after["llm_input_tokens"] - usage_before["llm_input_tokens"],
            llm_output_tokens=usage_after["llm_output_tokens"] - usage_before["llm_output_tokens"],
            embedding_input_tokens=(
                usage_after["embedding_input_tokens"] - usage_before["embedding_input_tokens"]
            ),
            confidence=round(confidence, 4),
            difficulty=case.get("difficulty", ""),
            tags=list(case.get("tags") or []),
            probe=case.get("probe", ""),
            error=error,
        ))
    return results


def _accuracy(items: Iterable[CaseResult], attr: str) -> float:
    seq = list(items)
    if not seq:
        return 0.0
    return round(sum(bool(getattr(item, attr)) for item in seq) / len(seq), 4)


def _macro_f1(
    items: Iterable[CaseResult],
    expected_attr: str,
    predicted_attr: str,
) -> tuple[float, Dict[str, Dict[str, float]]]:
    seq = list(items)
    labels = sorted({
        str(getattr(item, expected_attr)) for item in seq
    } | {
        str(getattr(item, predicted_attr)) for item in seq
    })
    if not labels:
        return 0.0, {}

    by_label: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(
            getattr(item, predicted_attr) == label and getattr(item, expected_attr) == label
            for item in seq
        )
        fp = sum(
            getattr(item, predicted_attr) == label and getattr(item, expected_attr) != label
            for item in seq
        )
        fn = sum(
            getattr(item, predicted_attr) != label and getattr(item, expected_attr) == label
            for item in seq
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = {
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return round(sum(metric["f1"] for metric in by_label.values()) / len(by_label), 4), by_label


def _p95_latency(items: Iterable[CaseResult]) -> float:
    values = sorted(item.latency_ms for item in items)
    if not values:
        return 0.0
    rank = max(0, int(len(values) * 0.95 + 0.999999) - 1)
    return values[rank]


def _estimated_cost_cny(items: Iterable[CaseResult], pricing: Dict[str, Optional[float]]) -> Optional[float]:
    seq = list(items)
    llm_input = sum(item.llm_input_tokens for item in seq)
    llm_output = sum(item.llm_output_tokens for item in seq)
    embedding_input = sum(item.embedding_input_tokens for item in seq)

    required_prices = (
        (llm_input and pricing["llm_input_cny_per_mtok"] is None)
        or (llm_output and pricing["llm_output_cny_per_mtok"] is None)
        or (embedding_input and pricing["embedding_input_cny_per_mtok"] is None)
    )
    if required_prices:
        return None

    total = (
        llm_input * (pricing["llm_input_cny_per_mtok"] or 0.0)
        + llm_output * (pricing["llm_output_cny_per_mtok"] or 0.0)
        + embedding_input * (pricing["embedding_input_cny_per_mtok"] or 0.0)
    ) / 1_000_000
    return round(total, 8)


def _quality_metrics(items: Iterable[CaseResult]) -> Dict[str, Any]:
    seq = list(items)
    intent_macro_f1, _ = _macro_f1(seq, "expected_intent", "predicted_intent")
    group_macro_f1, _ = _macro_f1(seq, "expected_intent_group", "predicted_intent_group")
    error_count = sum(1 for item in seq if item.error)
    return {
        "total": len(seq),
        "intent_accuracy": _accuracy(seq, "intent_ok"),
        "group_accuracy": _accuracy(seq, "group_ok"),
        "intent_macro_f1": intent_macro_f1,
        "group_macro_f1": group_macro_f1,
        "avg_latency_ms": round(sum(item.latency_ms for item in seq) / len(seq), 1) if seq else 0.0,
        "p95_latency_ms": _p95_latency(seq),
        "errors": error_count,
        "error_rate": round(error_count / len(seq), 4) if seq else 0.0,
    }


def summarize(
    results: List[CaseResult],
    pricing: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    by_strategy: Dict[str, List[CaseResult]] = defaultdict(list)
    for result in results:
        by_strategy[result.strategy].append(result)

    summary: Dict[str, Any] = {}
    for strategy, items in sorted(by_strategy.items()):
        difficulty = {}
        for level in sorted(set(item.difficulty for item in items)):
            bucket = [item for item in items if item.difficulty == level]
            difficulty[level] = _quality_metrics(bucket)

        tag_counter: Counter[str] = Counter(tag for item in items for tag in item.tags)
        weak_tags = {}
        for tag in sorted(tag_counter):
            bucket = [item for item in items if tag in item.tags]
            weak_tags[tag] = _quality_metrics(bucket)

        failures = [
            {
                "id": item.id,
                "message": item.message,
                "expected": {
                    "intent": item.expected_intent,
                    "group": item.expected_intent_group,
                },
                "predicted": {
                    "intent": item.predicted_intent,
                    "group": item.predicted_intent_group,
                },
                "difficulty": item.difficulty,
                "tags": item.tags,
                "probe": item.probe,
                "error": item.error,
            }
            for item in items
            if not (item.intent_ok and item.group_ok)
        ]

        intent_macro_f1, intent_by_label = _macro_f1(
            items,
            "expected_intent",
            "predicted_intent",
        )
        group_macro_f1, group_by_label = _macro_f1(
            items,
            "expected_intent_group",
            "predicted_intent_group",
        )
        estimated_total_cost = _estimated_cost_cny(items, pricing)
        metrics = _quality_metrics(items)
        token_usage = {
            "llm_input_tokens": sum(item.llm_input_tokens for item in items),
            "llm_output_tokens": sum(item.llm_output_tokens for item in items),
            "embedding_input_tokens": sum(item.embedding_input_tokens for item in items),
        }
        summary[strategy] = {
            **metrics,
            "intent_macro_f1": intent_macro_f1,
            "group_macro_f1": group_macro_f1,
            "intent_by_label": intent_by_label,
            "group_by_label": group_by_label,
            "token_usage": token_usage,
            "estimated_total_cost_cny": estimated_total_cost,
            "cost_per_case_cny": (
                round(estimated_total_cost / len(items), 8)
                if estimated_total_cost is not None and items else None
            ),
            "by_difficulty": difficulty,
            "by_tag": weak_tags,
            "failures": failures,
        }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\nStrategy comparison")
    print("strategy      total  intent   group    i_f1     g_f1    avg_ms  p95_ms  err_rate  cost/case")
    print("---------------------------------------------------------------------------------------")
    for strategy, stats in summary.items():
        cost = stats["cost_per_case_cny"]
        cost_text = f"{cost:.6f}" if cost is not None else "n/a"
        print(
            f"{strategy:<12} "
            f"{stats['total']:>5}  "
            f"{stats['intent_accuracy']:<7.2%} "
            f"{stats['group_accuracy']:<7.2%} "
            f"{stats['intent_macro_f1']:<7.2%} "
            f"{stats['group_macro_f1']:<7.2%} "
            f"{stats['avg_latency_ms']:>7.1f} "
            f"{stats['p95_latency_ms']:>7.1f} "
            f"{stats['error_rate']:<8.2%} "
            f"{cost_text:>9}"
        )

    print("\nLargest failure buckets")
    for strategy, stats in summary.items():
        hard = stats["by_difficulty"].get("hard", {})
        print(
            f"- {strategy}: hard intent={hard.get('intent_accuracy', 0):.2%}, "
            f"group={hard.get('group_accuracy', 0):.2%}, "
            f"intent_f1={hard.get('intent_macro_f1', 0):.2%}, "
            f"group_f1={hard.get('group_macro_f1', 0):.2%}, "
            f"failures={len(stats['failures'])}"
        )


def _env_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as ex:
        raise ValueError(f"{name} must be a number") from ex


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--strategies", default="fusion,llm,embedding,pattern")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("ANTHROPIC_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--llm-input-price-cny-per-mtok",
        type=float,
        default=_env_optional_float("SUPPORT_NEXUS_EVAL_LLM_INPUT_PRICE_CNY_PER_MTOK"),
        help="LLM input price in CNY per million tokens; enables cost estimation.",
    )
    parser.add_argument(
        "--llm-output-price-cny-per-mtok",
        type=float,
        default=_env_optional_float("SUPPORT_NEXUS_EVAL_LLM_OUTPUT_PRICE_CNY_PER_MTOK"),
        help="LLM output price in CNY per million tokens; enables cost estimation.",
    )
    parser.add_argument(
        "--embedding-price-cny-per-mtok",
        type=float,
        default=_env_optional_float("SUPPORT_NEXUS_EVAL_EMBEDDING_PRICE_CNY_PER_MTOK"),
        help="Embedding input price in CNY per million tokens; enables cost estimation.",
    )
    args = parser.parse_args()

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    unknown = sorted(set(strategies) - {"fusion", "llm", "embedding", "pattern"})
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")

    cases_path = pathlib.Path(args.cases)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    pricing = {
        "llm_input_cny_per_mtok": args.llm_input_price_cny_per_mtok,
        "llm_output_cny_per_mtok": args.llm_output_price_cny_per_mtok,
        "embedding_input_cny_per_mtok": args.embedding_price_cny_per_mtok,
    }

    all_results: List[CaseResult] = []
    for strategy in strategies:
        # Isolate caches so latency and token usage of one strategy cannot affect another.
        recognizer = IntentRecognizer(
            api_key=args.api_key or "local-only",
            base_url=args.base_url or None,
            model=args.model,
        )
        strategy_results = await evaluate_strategy(recognizer, cases, strategy)
        all_results.extend(strategy_results)

    report = {
        "cases_path": str(cases_path),
        "model": args.model,
        "base_url": args.base_url or None,
        "case_count": len(cases),
        "strategies": strategies,
        "pricing_cny_per_mtok": pricing,
        "strategy_cache_mode": "isolated_per_strategy",
        "embedding_backend": (
            "DashScope/OpenAI-compatible embeddings when DASHSCOPE_API_KEY is set; "
            "default model qwen3.7-text-embedding-flash with 1024 dimensions; "
            "otherwise local char n-gram hashing, 256 dimensions"
        ),
        "summary": summarize(all_results, pricing),
        "results": [asdict(item) for item in all_results],
    }

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(report["summary"])
    print(f"\nWrote report: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
