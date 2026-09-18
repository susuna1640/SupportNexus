import asyncio

from agents.agent_orchestrator import (
    APIAgent,
    AgentProfile,
    AgentResponse,
    AgentType,
    BillingAgent,
    ContextWindowUsage,
    GeneralAgent,
    OnboardingAgent,
    EnterpriseAgent,
    AgentOrchestrator,
    Request,
    ResponseComposer,
    RoutingDecision,
    TechnicalAgent,
    build_shared_rag_tools,
    model_context_window_tokens,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "调用 /v1/chat 返回 401，同时想确认企业版套餐权限",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.API_INTEGRATION,
        "intent_group": "api_integration",
        "intent_confidence": 0.92,
        "domain_scores": {"api": 0.92, "enterprise": 0.65},
        "entities": {"error_code": ["401"], "endpoint": ["/v1/chat"], "workspace_id": ["ws_prod_1234"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TechnicalAgent.profile.role
    assert TechnicalAgent.profile.workflow != BillingAgent.profile.workflow
    assert APIAgent.profile.role != TechnicalAgent.profile.role
    assert OnboardingAgent.profile.workflow != EnterpriseAgent.profile.workflow
    assert TechnicalAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_error_code" in APIAgent.profile.tool_scope
    assert "validate_api_request_shape" in APIAgent.profile.tool_scope
    assert "build_onboarding_checklist" in OnboardingAgent.profile.tool_scope
    assert "check_enterprise_access_fields" in EnterpriseAgent.profile.tool_scope
    assert "lookup_error_code" in TechnicalAgent.profile.tool_scope
    assert "check_billing_fields" in BillingAgent.profile.tool_scope


def test_context_window_defaults_to_selected_model_metadata(monkeypatch):
    monkeypatch.delenv("SUPPORT_NEXUS_MODEL_CONTEXT_WINDOWS", raising=False)
    assert model_context_window_tokens("deepseek-v4-pro") == 1_000_000
    assert model_context_window_tokens("claude-3-5-sonnet-20241022") == 200_000
    assert model_context_window_tokens("private-proxy-model") == 32_768

    monkeypatch.setenv("SUPPORT_NEXUS_MODEL_CONTEXT_WINDOWS", '{"private-proxy-model": 131072}')
    assert model_context_window_tokens("private-proxy-model") == 131_072


def test_parallel_context_usage_reports_the_highest_agent_percentage():
    usage = AgentOrchestrator._highest_context_usage([
        AgentResponse(AgentType.API, "", True, context_usage=ContextWindowUsage(5_000, 1_000_000)),
        AgentResponse(AgentType.ENTERPRISE, "", True, context_usage=ContextWindowUsage(12_000, 100_000)),
    ])

    assert usage == ContextWindowUsage(12_000, 100_000)


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    api_packet = APIAgent(FakeClient(), "test-model")._build_role_packet(req)
    onboarding_packet = OnboardingAgent(FakeClient(), "test-model")._build_role_packet(req)
    enterprise_packet = EnterpriseAgent(FakeClient(), "test-model")._build_role_packet(req)
    technical_packet = TechnicalAgent(FakeClient(), "test-model")._build_role_packet(req)
    billing_packet = BillingAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "api_debug_fields" in api_packet
    assert "onboarding_fields" in onboarding_packet
    assert "enterprise_fields" in enterprise_packet
    assert "diagnostic_fields" in technical_packet
    assert "verification_fields" in billing_packet
    assert len({general_packet, api_packet, onboarding_packet, enterprise_packet, technical_packet, billing_packet}) == 6


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.API, "先排查 API Key 是否过期。", True),
        AgentResponse(AgentType.ENTERPRISE, "请确认 workspace 权限和企业套餐。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先排查 API Key 是否过期。")
    assert "补充说明" in content
    assert "workspace 权限" in content


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    api_tools = set(APIAgent(FakeClient(), "test-model").get_tools())
    onboarding_tools = set(OnboardingAgent(FakeClient(), "test-model").get_tools())
    enterprise_tools = set(EnterpriseAgent(FakeClient(), "test-model").get_tools())
    technical_tools = set(TechnicalAgent(FakeClient(), "test-model").get_tools())
    billing_tools = set(BillingAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields"}
    assert api_tools == {"lookup_error_code", "validate_api_request_shape", "build_diagnostic_plan"}
    assert onboarding_tools == {"build_onboarding_checklist", "suggest_required_fields"}
    assert enterprise_tools == {"check_enterprise_access_fields", "suggest_required_fields"}
    assert technical_tools == {"lookup_error_code", "build_diagnostic_plan"}
    assert billing_tools == {"check_billing_fields", "compare_amounts"}
    assert not general_tools & api_tools
    assert not technical_tools & billing_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_optimized(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "API 接入指南", "content": "调试失败时记录 request_id"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    api = APIAgent(FakeClient(), "test-model")
    onboarding = OnboardingAgent(FakeClient(), "test-model")
    enterprise = EnterpriseAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    for agent in (general, api, onboarding, enterprise, technical, billing):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_intent_group_is_mapped_from_the_final_intent_not_llm_output():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.threshold = 0.5
    recognizer._embedding_enabled = True

    llm = recognizer._normalize_llm_intent({
        "intent": "subscription",
        # Simulate a stale or malformed provider response. This must not affect routing.
        "intent_group": "api_integration",
        "confidence": 0.9,
    })
    intent, _, source_scores = recognizer._vote(
        llm,
        {"intent": IntentCategory.OTHER, "confidence": 0.0},
        {"intent": IntentCategory.OTHER, "confidence": 0.0},
    )

    assert intent == IntentCategory.SUBSCRIPTION
    assert recognizer._intent_group(intent) == "billing"
    assert "intent_group" not in llm
    assert "group_score" not in source_scores
    assert llm["domain_scores"]["billing"] == 0.9


def test_three_way_fusion_keeps_multiple_business_domains():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True

    scores = recognizer._fuse_domain_scores(
        {"intent": IntentCategory.TECHNICAL_LOGIN, "domain_scores": {"technical": 0.9, "billing": 0.65}},
        {"intent": IntentCategory.PAYMENT_ISSUE, "domain_scores": {"technical": 0.5, "billing": 0.8}},
        {"intent": IntentCategory.PAYMENT_ISSUE, "domain_scores": {"technical": 0.5, "billing": 1.0}},
    )

    assert scores["technical"] == 0.78
    assert scores["billing"] == 0.715
    assert scores["api"] == 0.0


def test_saas_domain_scores_select_primary_and_supporting_agents():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(FakeClient(), "test-model")],
        AgentType.API: [APIAgent(FakeClient(), "test-model")],
        AgentType.ONBOARDING: [OnboardingAgent(FakeClient(), "test-model")],
        AgentType.ENTERPRISE: [EnterpriseAgent(FakeClient(), "test-model")],
        AgentType.TECHNICAL: [TechnicalAgent(FakeClient(), "test-model")],
        AgentType.BILLING: [BillingAgent(FakeClient(), "test-model")],
    }

    decision = orchestrator._route_decision(make_request(
        domain_scores={"api": 0.91, "enterprise": 0.66, "billing": 0.18},
    ))

    assert decision.primary_agent == AgentType.API
    assert decision.supporting_agents == [AgentType.ENTERPRISE]
    assert "fusion_domain_scores" in decision.reason


def test_low_fusion_domain_scores_fall_back_to_general_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(FakeClient(), "test-model")],
        AgentType.API: [APIAgent(FakeClient(), "test-model")],
        AgentType.ONBOARDING: [OnboardingAgent(FakeClient(), "test-model")],
        AgentType.ENTERPRISE: [EnterpriseAgent(FakeClient(), "test-model")],
        AgentType.TECHNICAL: [TechnicalAgent(FakeClient(), "test-model")],
        AgentType.BILLING: [BillingAgent(FakeClient(), "test-model")],
    }

    decision = orchestrator._route_decision(make_request(
        message="帮我写一首诗",
        intent=IntentCategory.OTHER,
        intent_group="other",
        domain_scores={"api": 0.12, "technical": 0.18},
    ))

    assert decision.primary_agent == AgentType.GENERAL
    assert decision.supporting_agents == []


def test_tool_input_validation_rejects_unknown_fields():
    agent = TechnicalAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_error_code"]

    try:
        agent._validate_tool_input(spec, {"error_code": "401", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_saas_api_tools_explain_business_errors_and_validate_request_shape():
    req = make_request(entities={"error_code": ["PLAN_LIMIT_EXCEEDED"], "endpoint": ["/v1/audit"]})
    agent = APIAgent(FakeClient(), "test-model")

    error_result = agent.get_tools()["lookup_error_code"].handler(
        req,
        {"error_code": "PLAN_LIMIT_EXCEEDED"},
    )
    shape_result = agent.get_tools()["validate_api_request_shape"].handler(
        req,
        {
            "method": "POST",
            "endpoint": "/v1/audit",
            "environment": "production",
            "has_auth_header": True,
        },
    )

    assert "套餐" in error_result["meaning"]
    assert "request_id" in shape_result["missing_fields"]
    assert shape_result["real_request_sent"] is False


def test_onboarding_and_enterprise_tools_return_structured_boundaries():
    onboarding = OnboardingAgent(FakeClient(), "test-model")
    enterprise = EnterpriseAgent(FakeClient(), "test-model")

    checklist = onboarding.get_tools()["build_onboarding_checklist"].handler(
        make_request(intent=IntentCategory.ONBOARDING),
        {"role": "developer", "target_feature": "webhook"},
    )
    access = enterprise.get_tools()["check_enterprise_access_fields"].handler(
        make_request(
            intent=IntentCategory.ENTERPRISE_MGMT,
            entities={"workspace_id": ["ws_prod_1234"]},
        ),
        {"requested_action": "contract_renewal", "requester_role": "admin"},
    )

    assert any("webhook" in step.lower() for step in checklist["checklist"])
    assert access["requires_manual_review"] is True
    assert "company_name" in access["missing_fields"]


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_error_code"
        input = {"error_code": "401"}

    class TextBlock:
        type = "text"
        text = "已根据 401 错误码给出排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_error_code"]
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_error_code",
        "build_diagnostic_plan",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_agent_compacts_memory_when_full_request_exceeds_context_budget():
    class TextBlock:
        type = "text"
        text = "压缩后的上下文已用于生成回答。"

    class ContextClient:
        def __init__(self):
            self.calls = []
            self.count_calls = []

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def count_tokens(self, **kwargs):
                self.owner.count_calls.append(kwargs)
                background = str(kwargs["messages"][0].get("content", ""))
                tokens = 30000 if "压缩前的上下文" in background else 100
                return type("TokenCount", (), {"input_tokens": tokens})()

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return type("Response", (), {"content": [TextBlock()]})()

        @property
        def messages(self):
            return self.Messages(self)

    class RefreshedContext:
        def to_prompt_text(self, **kwargs):
            return "[会话摘要]\n保留了关键事实和待办。\n\n[最近对话]\nuser: 最近消息"

    class Memory:
        RECENT_MESSAGES_TO_KEEP = 20
        MIN_RECENT_MESSAGES_TO_KEEP = 4
        CONTEXT_MESSAGE_MAX_CHARS = 6000

        def __init__(self):
            self.compaction_calls = []

        async def compact_session(self, user_id, conv_id, *, keep_recent_messages):
            self.compaction_calls.append((user_id, conv_id, keep_recent_messages))
            return True

        async def get_context(self, user_id, conv_id, query):
            return RefreshedContext()

    client = ContextClient()
    memory = Memory()
    agent = TechnicalAgent(client, "test-model", memory_manager=memory)
    req = make_request(
        context="压缩前的上下文 " + "x" * 100,
        intent=IntentCategory.TECHNICAL,
    )

    response = asyncio.run(agent.handle(req))

    assert response.success is True
    assert memory.compaction_calls == [("u1", "c1", 20)]
    assert len(client.count_calls) == 2
    assert response.context_usage is not None
    assert response.context_usage.input_tokens == 100
    assert response.context_usage.window_tokens == 32_768
    assert response.context_usage.estimated is False
    assert "[会话摘要]" in client.calls[0]["messages"][0]["content"]
    assert "[角色契约]" in client.count_calls[0]["system"]
    assert {tool["name"] for tool in client.count_calls[0]["tools"]} == {
        "lookup_error_code",
        "build_diagnostic_plan",
    }
