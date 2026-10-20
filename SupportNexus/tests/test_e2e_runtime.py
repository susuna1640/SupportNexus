import asyncio
from types import SimpleNamespace

from evaluation.evaluator import (
    E2EQualitySuite,
    EndToEndEvaluator,
    EvalReport,
    QualityScores,
    load_e2e_quality_suite,
)


class _Value:
    def __init__(self, value: str):
        self.value = value


class _FakeOrchestrator:
    async def run(self, _request):
        return SimpleNamespace(
            intent=_Value("greeting"),
            primary_agent=_Value("general"),
            supporting_agents=[],
            tools_used=[],
            response="你好，我在。请告诉我需要什么帮助。",
        )


class _FakeJudge:
    async def judge_e2e(self, *_args, **_kwargs):
        return QualityScores(1.0, 1.0, 1.0, 1.0, safety=1.0)

    async def judge_conversation(self, *_args, **_kwargs):
        return {
            "scores": {
                "context_retention": 1.0,
                "state_progression": 1.0,
                "consistency": 1.0,
                "correction_handling": 1.0,
                "resolution": 1.0,
            },
            "critical_errors": [],
            "judge_failed": False,
            "error": None,
        }


def _evaluator() -> EndToEndEvaluator:
    # 不构造真实 LLM 客户端，确保评测接线可在离线单元测试中覆盖。
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._orchestrator = _FakeOrchestrator()
    evaluator._judge = _FakeJudge()
    evaluator._history = []
    evaluator._baseline = None
    evaluator._baseline_path = None
    return evaluator


def test_formal_e2e_suite_loads_from_versioned_data() -> None:
    suite = load_e2e_quality_suite()

    assert suite.suite_id == "e2e_quality_v1"
    assert len(suite.cases) == 40
    assert suite.rubric["pass_policy"]["normalized_safety_min"] == 0.9


def test_formal_e2e_run_records_hard_assertions_and_safety() -> None:
    suite = E2EQualitySuite(
        suite_id="e2e_quality_v1",
        rubric=load_e2e_quality_suite().rubric,
        cases=[
            {
                "id": "unit_greeting",
                "case_type": "single_turn",
                "question": "你好",
                "difficulty": "easy",
                "risk": "low",
                "expected": {
                    "intent": "greeting",
                    "routing": {
                        "primary": "general",
                        "required_supporting": [],
                        "allowed_supporting": [],
                    },
                    "escalation": "not_required",
                    "tools": {"required_all": [], "required_any": [], "forbidden": []},
                    "reference_facts": ["普通问候无需索取账户信息"],
                    "must_include": ["礼貌回应"],
                    "must_not_include": ["虚构账户查询"],
                },
            }
        ],
    )

    report = asyncio.run(_evaluator().run_e2e_quality_suite(suite))

    assert report.suite_id == "e2e_quality_v1"
    assert report.total == report.passed == 1
    assert report.avg_scores["safety"] == 1.0
    assert report.results[0].metadata["assertions"]["intent"]["passed"] is True
    assert report.results[0].metadata["assertions"]["tools"]["passed"] is True


def test_hard_assertions_expose_wrong_route_and_forbidden_tool() -> None:
    result = SimpleNamespace(
        intent=_Value("greeting"),
        primary_agent=_Value("api"),
        supporting_agents=[],
        tools_used=["lookup_error_code"],
        response="你好。",
    )
    expected = {
        "intent": "greeting",
        "routing": {"primary": "general", "required_supporting": [], "allowed_supporting": []},
        "escalation": "not_required",
        "tools": {"required_all": [], "required_any": [], "forbidden": ["lookup_error_code"]},
    }

    checks = EndToEndEvaluator._deterministic_checks(result, expected)

    assert checks["intent"]["passed"] is True
    assert checks["primary_agent"]["passed"] is False
    assert checks["tools"]["passed"] is False
    assert checks["tools"]["forbidden_used"] == ["lookup_error_code"]


def test_eval_api_defaults_to_formal_e2e_suite() -> None:
    from api import main as api_main

    class _FakeApiEvaluator:
        seen_suite_id = ""
        seen_cases = 0

        async def run_e2e_quality_suite(self, suite):
            self.seen_suite_id = suite.suite_id
            self.seen_cases = len(suite.cases)
            return EvalReport(
                timestamp="2026-09-19T00:00:00",
                total=1,
                passed=1,
                pass_rate=1.0,
                avg_scores={"safety": 1.0},
                regressions=[],
                recommendations=[],
                results=[],
                suite_id=suite.suite_id,
            )

    original = api_main._evaluator
    fake = _FakeApiEvaluator()
    api_main._evaluator = fake
    try:
        response = asyncio.run(api_main.run_eval())
    finally:
        api_main._evaluator = original

    assert fake.seen_suite_id == "e2e_quality_v1"
    assert fake.seen_cases == 40
    assert response["suite_id"] == "e2e_quality_v1"
