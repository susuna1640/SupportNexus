import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
CASES_PATH = DATA_DIR / "e2e_quality_cases_v1.json"
RUBRIC_PATH = DATA_DIR / "e2e_quality_rubric_v1.json"

INTENTS = {
    "query", "complaint", "request", "greeting", "escalation", "technical",
    "billing", "account", "feedback", "api_integration", "feature_inquiry",
    "onboarding", "enterprise_mgmt", "subscription", "order_status",
    "logistics", "refund", "invoice", "payment_issue", "account_security",
    "technical_login", "technical_crash", "human_handoff", "other",
}
AGENTS = {"general", "api", "onboarding", "enterprise", "technical", "billing", "escalation"}
TOOLS = {
    "inspect_request_context", "suggest_required_fields", "lookup_error_code",
    "validate_api_request_shape", "build_diagnostic_plan",
    "build_onboarding_checklist", "check_billing_fields", "compare_amounts",
    "check_enterprise_access_fields", "create_handoff_summary",
    "search_knowledge_base",
}


def _assert_expected(expected: dict) -> None:
    assert expected["intent"] in INTENTS
    for intent in expected.get("allowed_intents", []):
        assert intent in INTENTS

    routing = expected["routing"]
    assert routing["primary"] in AGENTS
    assert set(routing["required_supporting"]).issubset(AGENTS)
    assert set(routing["allowed_supporting"]).issubset(AGENTS)
    assert set(routing["required_supporting"]).issubset(routing["allowed_supporting"])

    assert expected["escalation"] in {"required", "not_required", "allowed"}
    tool_rules = expected["tools"]
    for key in ("required_all", "required_any", "forbidden"):
        assert set(tool_rules[key]).issubset(TOOLS)

    assert expected["reference_facts"]
    assert expected["must_include"]
    assert isinstance(expected["must_not_include"], list)


def test_e2e_quality_dataset_is_well_formed() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert len(cases) >= 30

    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    single_turn = [case for case in cases if case["case_type"] == "single_turn"]
    multi_turn = [case for case in cases if case["case_type"] == "multi_turn"]
    assert len(single_turn) >= 20
    assert len(multi_turn) >= 8

    for case in single_turn:
        assert case["question"].strip()
        _assert_expected(case["expected"])

    for case in multi_turn:
        assert len(case["turns"]) >= 2
        assert len(case["turns"]) == len(case["turn_expectations"])
        assert case["conversation_expectations"]
        for index, expected in enumerate(case["turn_expectations"]):
            assert expected["turn"] == index
            _assert_expected(expected)


def test_e2e_quality_rubric_has_required_dimensions() -> None:
    rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    assert rubric["version"] == "1.0.0"
    assert set(rubric["dimensions"]) == {
        "relevance", "accuracy", "completeness", "helpfulness", "safety"
    }
    assert rubric["pass_policy"]["no_critical_error"] is True
    assert rubric["pass_policy"]["judge_error_is_pass"] is False
