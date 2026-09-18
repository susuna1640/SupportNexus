"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 技术排障建议
  - 账单字段核验
  - 共享知识库 RAG

订单查询、退款执行、账单修改等需要真实业务系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "intent_confidence": round(req.intent_confidence, 4),
        "domain_scores": req.domain_scores or {},
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent == "api_integration":
        fields = ["接口路径", "HTTP 状态码或业务错误码", "request_id", "调用环境"]
    elif intent in {"feature_inquiry", "onboarding"}:
        fields = ["使用角色", "目标功能", "当前接入阶段"]
    elif intent == "enterprise_mgmt":
        fields = ["workspace_id", "当前角色", "目标权限或企业功能"]
    elif intent == "subscription":
        fields = ["workspace_id", "套餐名称", "账单周期或用量范围"]
    elif intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码的排查方向，不声称读取了服务端日志。"""
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "429": ("请求触发限流", ["确认当前套餐 QPS/并发/月度额度", "使用指数退避重试", "评估是否需要提升套餐或申请限额"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
        "AUTH_001": ("API Key 无效或未携带", ["检查 Authorization 请求头格式", "确认 API Key 属于当前 workspace", "不要在公开渠道发送完整密钥"]),
        "AUTH_002": ("签名或时间戳校验失败", ["确认使用正确 secret 计算签名", "检查服务器时间偏差", "确认 sandbox/production 环境未混用"]),
        "PLAN_LIMIT_EXCEEDED": ("套餐额度或功能权限不足", ["核对当前套餐包含的接口和额度", "确认 workspace 是否开启目标功能", "需要变更套餐时转订阅或企业支持"]),
        "RATE_LIMIT_EXCEEDED": ("调用频率超过限制", ["降低并发或加入退避策略", "查看当前 QPS 和月度调用量", "高峰流量需要评估限额提升"]),
        "WEBHOOK_SIGNATURE_INVALID": ("Webhook 签名校验失败", ["使用原始请求体计算签名", "检查 timestamp 是否在允许窗口内", "确认 webhook secret 来自当前环境"]),
        "INVALID_REQUEST_SCHEMA": ("请求体结构不符合接口要求", ["核对必填字段和字段类型", "确认 SDK/API 版本", "用最小请求复现并记录 request_id"]),
        "WORKSPACE_NOT_FOUND": ("workspace 不存在或不可访问", ["确认 workspace_id 是否正确", "确认调用方账号是否属于该 workspace", "检查企业权限或 SSO 同步状态"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def validate_api_request_shape(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """API 工具：检查一次接口调试所需字段是否齐全，不发送真实请求。"""
    method = str(args.get("method", "")).upper().strip()
    endpoint = str(args.get("endpoint", "")).strip()
    environment = str(args.get("environment", "")).strip().lower()
    has_auth_header = bool(args.get("has_auth_header", False))

    missing_fields = []
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        missing_fields.append("method")
    if not endpoint.startswith("/v"):
        missing_fields.append("endpoint")
    if environment not in {"sandbox", "production", "staging", "test"}:
        missing_fields.append("environment")
    if not has_auth_header:
        missing_fields.append("authorization_header")
    if not req.entities.get("request_id"):
        missing_fields.append("request_id")

    return {
        "success": True,
        "method": method or None,
        "endpoint": endpoint or None,
        "environment": environment or None,
        "missing_fields": missing_fields,
        "safe_debug_next_step": (
            "字段齐全，可基于状态码、响应体摘要和 request_id 继续排查"
            if not missing_fields else "先补齐缺失字段，再判断认证、权限、限流或请求体问题"
        ),
        "real_request_sent": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险排障顺序。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """订阅账单工具：检查必要核验字段是否齐全。"""
    fields = {
        "workspace_id": bool(req.entities.get("workspace_id")),
        "account_id": bool(req.entities.get("account_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
        "billing_period": bool(args.get("billing_period")),
        "plan_name": bool(args.get("plan_name")),
    }
    return {
        "fields": fields,
        "missing_fields": [
            name for name, present in fields.items()
            if not present and name not in {"account_id"}
        ],
        "can_confirm_refund": False,
        "reason": "当前工具只做订阅账单字段检查，不连接真实支付、合同或订阅系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只做用户明确提供金额之间的算术。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def build_onboarding_checklist(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """Onboarding 工具：生成 SaaS 首次接入检查清单。"""
    role = str(args.get("role", "developer")).lower().strip()
    target = str(args.get("target_feature", "api_integration")).lower().strip()
    base_steps = [
        "创建 workspace 并确认管理员",
        "生成仅服务端使用的 API Key",
        "完成 sandbox 环境最小请求",
        "配置错误监控和 request_id 日志",
    ]
    if target in {"webhook", "webhooks"}:
        base_steps.extend(["配置 webhook 回调地址", "校验签名和重试策略"])
    elif target in {"sso", "scim"}:
        base_steps.extend(["确认企业套餐权限", "准备 IdP 元数据并联系企业管理员"])
    else:
        base_steps.extend(["接入生产环境", "完成上线前流量和权限检查"])

    return {
        "success": True,
        "role": role,
        "target_feature": target,
        "checklist": base_steps,
        "requires_configuration_review": target in {"sso", "scim"},
    }


def check_enterprise_access_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """企业支持工具：判断企业权限/合同类请求需要哪些核验字段。"""
    requested_action = str(args.get("requested_action", "unknown")).lower().strip()
    has_workspace = bool(req.entities.get("workspace_id"))
    has_account = bool(req.entities.get("account_id"))
    role = str(args.get("requester_role", "")).lower().strip()

    high_risk_actions = {"admin_change", "sso_config", "scim_sync", "contract_renewal", "discount", "ip_allowlist"}
    required_fields = ["workspace_id", "requester_role"]
    if requested_action in high_risk_actions:
        required_fields.extend(["admin_approval", "business_reason"])
    if requested_action in {"contract_renewal", "discount"}:
        required_fields.extend(["company_name", "renewal_period"])

    known = {
        "workspace_id": has_workspace,
        "account_id": has_account,
        "requester_role": bool(role),
        "admin_approval": bool(args.get("admin_approval")),
        "business_reason": bool(args.get("business_reason")),
        "company_name": bool(args.get("company_name")),
        "renewal_period": bool(args.get("renewal_period")),
    }
    return {
        "success": True,
        "requested_action": requested_action,
        "known_fields": known,
        "required_fields": required_fields,
        "missing_fields": [field for field in required_fields if not known.get(field, False)],
        "requires_manual_review": requested_action in high_risk_actions,
        "reason": "企业权限、SSO/SCIM、合同和折扣类请求必须经过管理员或已授权流程核验",
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_optimized(
            "knowledge_hybrid_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、技术、账单和升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释常见 HTTP 和 SaaS 业务错误码的可能含义和低风险排查方向；不会读取服务端日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、500、AUTH_001"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据运行环境和是否可复现生成排障顺序，不执行修改配置等操作。",
            {
                "environment": {"type": "string", "description": "App、浏览器、服务端或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def api_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释 API/HTTP/SaaS 业务错误码，返回含义、排查步骤和安全边界；不会读取后台日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、429、AUTH_001、PLAN_LIMIT_EXCEEDED"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "validate_api_request_shape": make_tool(
            "validate_api_request_shape",
            "检查接口调试所需字段是否齐全，不发送真实请求，不读取服务端日志。",
            {
                "method": {"type": "string", "description": "HTTP 方法，例如 GET、POST"},
                "endpoint": {"type": "string", "description": "接口路径，例如 /v1/chat"},
                "environment": {"type": "string", "description": "sandbox、production、staging 或 test"},
                "has_auth_header": {"type": "boolean", "description": "是否已携带 Authorization 请求头"},
            },
            validate_api_request_shape,
            required=["method", "endpoint", "environment", "has_auth_header"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据运行环境和是否可复现生成排障顺序，不执行修改配置等操作。",
            {
                "environment": {"type": "string", "description": "SDK、服务端、Webhook、控制台或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def onboarding_tools() -> Dict[str, AgentToolSpec]:
    return {
        "build_onboarding_checklist": make_tool(
            "build_onboarding_checklist",
            "根据用户角色和目标功能生成首次接入检查清单。",
            {
                "role": {"type": "string", "description": "用户角色，例如 developer、admin、cs、pm"},
                "target_feature": {"type": "string", "description": "目标功能，例如 api_integration、webhook、sso"},
            },
            build_onboarding_checklist,
            required=["role", "target_feature"],
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields",
            "检查 SaaS 订阅账单核验字段是否齐全；不连接真实支付、合同或订阅系统。",
            {
                "payment_channel": {"type": "string", "description": "支付渠道，例如银行卡、企业转账、Stripe"},
                "billing_period": {"type": "string", "description": "账单周期，例如 2026-09"},
                "plan_name": {"type": "string", "description": "套餐名称，例如 Starter、Pro、Enterprise"},
            },
            check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否重复扣款，也不执行退款或账单调整。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }


def enterprise_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_enterprise_access_fields": make_tool(
            "check_enterprise_access_fields",
            "检查企业权限、SSO/SCIM、审计日志、合同续费等请求需要哪些核验字段；不执行真实变更。",
            {
                "requested_action": {
                    "type": "string",
                    "description": "例如 admin_change、sso_config、scim_sync、contract_renewal、audit_log",
                },
                "requester_role": {"type": "string", "description": "请求者角色，例如 owner、admin、member、unknown"},
                "admin_approval": {"type": "boolean", "description": "是否已有管理员批准"},
                "business_reason": {"type": "string", "description": "业务理由或变更背景"},
                "company_name": {"type": "string", "description": "企业名称，合同/续费场景可选"},
                "renewal_period": {"type": "string", "description": "续费周期，合同/续费场景可选"},
            },
            check_enterprise_access_fields,
            required=["requested_action", "requester_role"],
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }
