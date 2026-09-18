"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略：
  1. 三路融合输出业务领域分数
  2. 分数最高的领域为主 Agent，超过阈值的其他领域为辅助 Agent
  3. 未命中业务阈值或专属 Agent 不可用时，降级到 GeneralAgent

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

"""
import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    api_tools,
    build_shared_rag_tools,
    billing_tools,
    enterprise_tools,
    general_tools,
    onboarding_tools,
    technical_tools,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer
from core.llm_utils import extract_text_content
from core.observability import capture, observation, update, usage_details

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # 通用客服
    API       = "api"        # API 接入支持
    ONBOARDING = "onboarding" # 新用户引导
    ENTERPRISE = "enterprise" # 企业账户
    TECHNICAL = "technical"  # 技术支持
    BILLING   = "billing"    # 订阅/账单


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    boundaries: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；错误配置不应阻塞服务启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


def _agent_max_tokens(agent_name: str, default: int = 2048) -> int:
    """读取 Agent 输出预算，支持全局与单个 Agent 两级覆盖。"""
    global_default = _env_int("SUPPORT_NEXUS_AGENT_MAX_TOKENS", default)
    value = _env_int(f"SUPPORT_NEXUS_{agent_name.upper()}_MAX_TOKENS", global_default)
    return max(256, value)


_BUILTIN_MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    # Anthropic 默认模型。
    "claude-3-5-sonnet-20241022": 200_000,
    # 当前项目默认使用的 DeepSeek Anthropic-compatible 模型与版本别名。
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-pro-0813": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-flash-0731": 1_000_000,
}


def _optional_positive_int(name: str) -> Optional[int]:
    """读取显式的正整数覆盖；未配置时返回 None。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("忽略非法整数配置 %s=%r", name, raw)
        return None
    if value <= 0:
        logger.warning("忽略非正整数配置 %s=%r", name, raw)
        return None
    return value


def model_context_window_tokens(model: str) -> int:
    """从可维护的模型目录解析上下文窗口；未知模型保守回退到 32K。"""
    normalized = (model or "").strip().lower()
    configured = os.getenv("SUPPORT_NEXUS_MODEL_CONTEXT_WINDOWS", "").strip()
    if configured:
        try:
            catalog = json.loads(configured)
            if isinstance(catalog, dict):
                for catalog_model, window in catalog.items():
                    if str(catalog_model).strip().lower() == normalized:
                        value = int(window)
                        if value > 0:
                            return value
                        logger.warning("忽略非正模型窗口: %s=%r", catalog_model, window)
                        break
            else:
                logger.warning("忽略 SUPPORT_NEXUS_MODEL_CONTEXT_WINDOWS：必须是 JSON 对象")
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("忽略非法 SUPPORT_NEXUS_MODEL_CONTEXT_WINDOWS JSON")

    return _BUILTIN_MODEL_CONTEXT_WINDOWS.get(normalized, 32_768)


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass(frozen=True)
class ContextWindowUsage:
    """某个业务 Agent 首次生成前的输入窗口占用。"""
    input_tokens: int
    window_tokens: int
    estimated: bool = False

    @property
    def percent(self) -> float:
        return self.input_tokens / self.window_tokens * 100 if self.window_tokens else 0.0


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    context_usage: Optional[ContextWindowUsage] = None


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    domain_scores: Dict[str, float] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    context_usage: Optional[ContextWindowUsage] = None


@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用、角色契约和统计。"""

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
        memory_manager: Optional[Any] = None,
    ):
        self._client = client
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._skill_manager = skill_manager
        self._memory_manager = memory_manager
        self.stats   = AgentStats()
        self._last_tools_used: List[str] = []
        self._last_tool_traces: List[Dict[str, Any]] = []
        self._shared_tools: Dict[str, AgentToolSpec] = {}

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        """返回该角色真实可调用的工具白名单。"""
        return dict(self._shared_tools)

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    def set_memory_manager(self, memory_manager: Optional[Any]) -> None:
        self._memory_manager = memory_manager

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        self._last_tools_used = []
        self._last_tool_traces = []
        context_usage: Optional[ContextWindowUsage] = None
        try:
            content, context_usage = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                tools_used=list(self._last_tools_used),
                tool_traces=list(self._last_tool_traces),
                context_usage=context_usage,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tool_traces=list(self._last_tool_traces),
                context_usage=context_usage,
            )

    async def _call_llm(self, req: Request) -> Tuple[str, Optional[ContextWindowUsage]]:
        messages = self._build_messages(req)
        tools = self._tools_for_request()
        tool_payloads = self._tool_payloads(tools)
        context_changed, context_usage = await self._compact_context_if_needed(req, messages, tool_payloads)
        if context_changed:
            messages = self._build_messages(req)

        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        for attempt in range(3):
            request_kwargs: Dict[str, Any] = {
                "model": self._model,
                "max_tokens": self.profile.max_tokens,
                "temperature": self.profile.temperature,
                "system": self._build_system_prompt(req),
                "messages": messages,
            }
            if tools:
                request_kwargs["tools"] = tool_payloads
            with observation(
                f"{self.agent_type.value}-agent-generation",
                as_type="generation",
                model=self._model,
                input={
                    "round": attempt + 1,
                    "system": capture(request_kwargs["system"]),
                    "messages": capture(messages),
                },
            ) as generation:
                try:
                    resp = await self._client.messages.create(**request_kwargs)
                except Exception as ex:
                    update(generation, output={"error": capture(str(ex))})
                    raise
            tool_uses = [block for block in (resp.content or []) if self._block_type(block) == "tool_use"]
            update(
                generation,
                output={
                    "text": capture(extract_text_content(resp.content)),
                    "tool_calls": [
                        {
                            "name": self._block_value(block, "name"),
                            "input": capture(self._block_value(block, "input") or {}),
                        }
                        for block in tool_uses
                    ],
                    "stop_reason": getattr(resp, "stop_reason", None),
                },
                usage_details=usage_details(resp),
            )
            if not tool_uses:
                self._last_tools_used = tools_used
                return extract_text_content(resp.content), context_usage

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in tool_uses:
                name = self._block_value(block, "name")
                tool_use_id = self._block_value(block, "id")
                args = self._block_value(block, "input") or {}
                spec = tools.get(name)
                tool_t0 = time.monotonic()
                call_success = True
                result_success: Optional[bool] = None
                error_text = ""
                with observation(
                    f"{self.agent_type.value}-tool-{name}",
                    as_type="tool",
                    input={"arguments": capture(args)},
                ) as tool_observation:
                    if spec is None:
                        call_success = False
                        result = {"success": False, "error": f"工具不在 {self.agent_type.value} Agent 白名单中"}
                        error_text = result["error"]
                    else:
                        try:
                            self._validate_tool_input(spec, args)
                            result = spec.handler(req, args)
                            if inspect.isawaitable(result):
                                result = await result
                            tools_used.append(name)
                            if isinstance(result, dict) and "success" in result:
                                result_success = bool(result.get("success"))
                        except Exception as ex:
                            call_success = False
                            logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                            error_text = str(ex)
                            result = {"success": False, "error": error_text}
                tool_latency_ms = (time.monotonic() - tool_t0) * 1000
                if not error_text and isinstance(result, dict):
                    error_text = str(result.get("error", "") or "")
                update(
                    tool_observation,
                    output={
                        "success": call_success,
                        "result_success": result_success,
                        "result": capture(result),
                        "error": capture(error_text) if error_text else "",
                    },
                )
                tool_traces.append(
                    {
                        "agent_type": self.agent_type.value,
                        "tool_name": name,
                        "tool_use_id": tool_use_id,
                        "input": dict(args),
                        "success": call_success,
                        "result_success": result_success,
                        "latency_ms": round(tool_latency_ms, 1),
                        "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                        "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                        "error": error_text,
                    }
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": self._tool_result_content(result),
                })
            messages.append({"role": "user", "content": tool_results})

        self._last_tools_used = tools_used
        self._last_tool_traces = tool_traces
        raise RuntimeError(f"{self.agent_type.value} 工具调用超过最大轮数")

    def _tools_for_request(self) -> Dict[str, AgentToolSpec]:
        """返回业务工具，并在有 Skill 目录时增加受角色约束的正文加载工具。"""
        tools = self.get_tools()
        if self._skill_manager is None or not self._skill_manager.catalog_for(self.agent_type.value):
            return tools

        loaded_skill_ids: set[str] = set()

        def load_agent_skill(_req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
            skill_id = str(args.get("skill_id", "")).strip().lower()
            if skill_id in loaded_skill_ids:
                return {"success": True, "skill_id": skill_id, "status": "already_loaded"}
            if len(loaded_skill_ids) >= 2:
                return {
                    "success": False,
                    "error": "单次请求最多加载两份 Skill；请基于已加载规则继续处理",
                }
            result = self._skill_manager.load_for_agent(skill_id, self.agent_type.value)
            if result.get("success"):
                loaded_skill_ids.add(skill_id)
            return result

        tools["load_agent_skill"] = AgentToolSpec(
            name="load_agent_skill",
            description="读取当前 Agent 目录中一个 Skill 的详细 SOP；仅在需要细分处理流程、核验字段或升级规则时调用。",
            input_schema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "系统 Prompt 中列出的 Skill ID",
                    },
                },
                "required": ["skill_id"],
                "additionalProperties": False,
            },
            handler=load_agent_skill,
        )
        return tools

    @staticmethod
    def _clean_text(value: str) -> str:
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _truncate_for_prompt(cls, value: str, max_chars: int, marker: str) -> str:
        """仅收缩送入模型的副本，不改写原始请求或记忆。"""
        text = cls._clean_text(value)
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        half = max(1, (max_chars - len(marker) - 2) // 2)
        return f"{text[:half]}\n{marker}\n{text[-half:]}"

    def _build_messages(self, req: Request) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{self._clean_text(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[结构化实体]\n{self._clean_text(entities_text)}"})
            messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})
        role_packet = self._build_role_packet(req)
        if role_packet:
            messages.append({"role": "user", "content": f"[角色输入契约]\n{self._clean_text(role_packet)}"})
            messages.append({"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"})
        current_message = self._truncate_for_prompt(
            req.message,
            _env_int("SUPPORT_NEXUS_CURRENT_MESSAGE_MAX_CHARS", 12000),
            "...[单条用户消息已截断]...",
        )
        messages.append({"role": "user", "content": current_message})
        return messages

    @staticmethod
    def _tool_payloads(tools: Dict[str, AgentToolSpec]) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in tools.values()
        ]

    @staticmethod
    def _tool_result_content(result: Any) -> str:
        content = json.dumps(result, ensure_ascii=False)
        max_chars = _env_int("SUPPORT_NEXUS_TOOL_RESULT_MAX_CHARS", 6000)
        if len(content) <= max_chars:
            return content
        half = max(1, (max_chars - 40) // 2)
        return f"{content[:half]}\n...[工具结果已截断]...\n{content[-half:]}"

    async def _compact_context_if_needed(
        self,
        req: Request,
        messages: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[ContextWindowUsage]]:
        """按当前 Agent 的模型窗口预检，并在必要时压缩当前会话。"""
        if self._memory_manager is None:
            return False, None

        context_window = (
            _optional_positive_int(f"SUPPORT_NEXUS_{self.agent_type.value.upper()}_CONTEXT_WINDOW_TOKENS")
            or _optional_positive_int("SUPPORT_NEXUS_CONTEXT_WINDOW_TOKENS")
            or model_context_window_tokens(self._model)
        )
        safety_tokens = _env_int("SUPPORT_NEXUS_CONTEXT_SAFETY_TOKENS", 1024)
        threshold = _env_float("SUPPORT_NEXUS_CONTEXT_COMPACTION_RATIO", 0.70)
        available_input = max(1, context_window - self.profile.max_tokens - safety_tokens)
        budget = max(1, int(available_input * min(max(threshold, 0.1), 0.95)))
        keep_recent = getattr(self._memory_manager, "RECENT_MESSAGES_TO_KEEP", 20)
        min_recent = getattr(self._memory_manager, "MIN_RECENT_MESSAGES_TO_KEEP", 4)
        current_messages = messages
        context_changed = False

        while True:
            input_tokens, estimated = await self._count_input_tokens(req, current_messages, tool_payloads)
            if input_tokens <= budget:
                return context_changed, ContextWindowUsage(input_tokens, context_window, estimated)

            logger.info(
                "%s Agent 输入 %s tokens 超过压缩阈值 %s，压缩会话 %s/%s（保留 %s 条原文）",
                self.agent_type.value,
                input_tokens,
                budget,
                req.user_id,
                req.conv_id,
                keep_recent,
            )
            compacted = await self._memory_manager.compact_session(
                req.user_id,
                req.conv_id,
                keep_recent_messages=keep_recent,
            )
            context = await self._memory_manager.get_context(req.user_id, req.conv_id, query=req.message)
            refreshed_context = context.to_prompt_text(
                max_message_chars=getattr(self._memory_manager, "CONTEXT_MESSAGE_MAX_CHARS", 6000),
            )
            if refreshed_context != req.context:
                req.context = refreshed_context
            refreshed_messages = self._build_messages(req)
            if refreshed_messages != current_messages:
                context_changed = True
                current_messages = refreshed_messages
                continue

            if compacted:
                # 正常情况下压缩必然改变 prompt；这里避免异常存储实现造成空转。
                logger.warning(
                    "%s Agent 已压缩会话但 prompt 未变化，保留当前上下文继续执行",
                    self.agent_type.value,
                )
                return context_changed, ContextWindowUsage(input_tokens, context_window, estimated)

            if keep_recent <= min_recent:
                logger.warning(
                    "%s Agent 输入仍超过压缩阈值；保留当前消息并继续执行",
                    self.agent_type.value,
                )
                return context_changed, ContextWindowUsage(input_tokens, context_window, estimated)
            keep_recent = max(min_recent, keep_recent - 2)

    async def _count_input_tokens(
        self,
        req: Request,
        messages: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
    ) -> Tuple[int, bool]:
        system = self._build_system_prompt(req)
        counter = getattr(getattr(self._client, "messages", None), "count_tokens", None)
        if counter is not None:
            try:
                count_kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "system": system,
                    "messages": messages,
                }
                if tool_payloads:
                    count_kwargs["tools"] = tool_payloads
                result = await counter(**count_kwargs)
                return int(getattr(result, "input_tokens", 0)), False
            except Exception as ex:
                logger.warning("%s Agent token 预检失败，使用保守估算: %s", self.agent_type.value, ex)

        serialized = json.dumps(
            {"system": system, "messages": messages, "tools": tool_payloads},
            ensure_ascii=False,
            default=str,
        )
        # 不同兼容服务的 tokenizer 未必可得；按字符数上界估算会较早压缩，
        # 但不会因中文等高 token 密度文本而低估输入窗口。
        return max(1, len(serialized)), True

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _validate_tool_input(spec: AgentToolSpec, args: Any) -> None:
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        schema = spec.input_schema
        for field_name in schema.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺少必需参数: {field_name}")
        properties = schema.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise ValueError(f"参数 {key} 类型错误，期望 {expected}")

    def _build_system_prompt(self, req: Request) -> str:
        """构建角色契约与仅含摘要的、按需 Skill 目录。"""
        profile_prompt = (
            f"\n\n[角色契约]\n"
            f"角色：{self.profile.role}\n"
            f"职责：{self.profile.mission}\n"
            f"处理流程：{' -> '.join(self.profile.workflow)}\n"
            f"可用输入：{'；'.join(self.profile.input_contract)}\n"
            f"输出要求：{'；'.join(self.profile.output_contract)}\n"
            f"处理边界：{'；'.join(self.profile.boundaries) or '按通用客服规则处理'}\n"
            f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
            "不要声称执行了未提供的后台查询、订阅变更、权限调整、合同处理或退款操作；缺少证据时明确说明需要核验。"
        )
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is None:
            return base_prompt
        catalog_prompt = self._skill_manager.catalog_prompt_for(self.agent_type.value)
        if not catalog_prompt:
            return base_prompt
        return f"{base_prompt}\n\n{catalog_prompt}"

    def _build_role_packet(self, req: Request) -> str:
        """给子 Agent 的确定性输入包；子类可补充领域字段。"""
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "intent_confidence": round(req.intent_confidence, 4),
            "domain_score": round(req.domain_scores.get(self.agent_type.value, 0.0), 4),
            "available_entities": req.entities or {},
        }
        return json.dumps(packet, ensure_ascii=False)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="通用客服分诊与首轮接待",
        mission="快速回答基础问题，澄清不完整需求，并将明确的业务问题交给对应专业 Agent。",
        workflow=("复述诉求", "判断业务范围", "直接回答或补充必要信息", "给出下一步"),
        input_contract=("对话历史", "长期用户记忆", "意图与领域分数", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        boundaries=("不执行权限、资金或隐私相关的后台操作", "不声称已经创建工单或完成转接"),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
        temperature=0.3,
        max_tokens=_agent_max_tokens("general"),
    )
    system_prompt = """
你是 SupportNexus 的通用客服分诊员，负责首轮接待、基础咨询、信息澄清和跨领域问题拆分。

你可以：
- 直接回答已知的基础产品咨询；需要事实依据时调用 `search_knowledge_base` 检索知识库。
- 调用 `inspect_request_context` 查看当前请求已识别的意图、实体和上下文。
- 调用 `suggest_required_fields`，只追问推进处理所必需的字段。
- 当系统提供 Skill 目录且需要具体分诊/升级 SOP 时，调用 `load_agent_skill` 读取自己的规则。

处理方式：先用一句话回应或复述核心诉求；复合问题按 API、上手、企业、技术、账单等主题拆开说明。信息不足时，一次只问最关键的 1–3 项，不要把所有可能字段一次性抛给用户。

你不能查询真实订单、账户、支付或工单系统；不能承诺已创建工单、已转接人工或已完成任何后台操作。涉及资金、权限、安全或明确专业问题时，说明需要相应流程核验，而不是编造处理结果。
""".strip()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = ["api", "onboarding", "enterprise", "technical", "billing"]
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class APIAgent(BaseAgent):
    agent_type = AgentType.API
    profile = AgentProfile(
        role="API 接入与开发者支持",
        mission="基于 API 文档、错误码、request_id 和调用上下文，帮助开发者完成接口接入、鉴权、Webhook、SDK 与限流排查。",
        workflow=("确认接口与环境", "检查鉴权和权限", "解释错误码", "给出最小复现/调试步骤", "说明核验边界"),
        input_contract=("接口路径", "HTTP 方法", "错误码", "request_id", "workspace_id", "SDK/语言版本", "知识库上下文"),
        output_contract=("问题复述", "可能原因", "可验证排查步骤", "需要补充的调试字段", "安全边界"),
        boundaries=("不得要求完整 API Key、Token 或 webhook secret", "不得声称已查询后台日志或修改套餐权限"),
        tool_scope=("search_knowledge_base", "lookup_error_code", "validate_api_request_shape", "build_diagnostic_plan"),
        temperature=0.1,
        max_tokens=_agent_max_tokens("api"),
    )
    system_prompt = """
你是 SupportNexus 的 B2B SaaS API 支持专家，处理接口接入、鉴权、错误码、SDK、Webhook、限流和 request_id 排查。

你可以：
- 调用 `search_knowledge_base` 查询 API 文档、错误说明和接入规范。
- 调用 `lookup_error_code` 解释 HTTP/业务错误码及低风险排查方向。
- 调用 `validate_api_request_shape` 检查 method、endpoint、环境、鉴权头和 request_id 是否齐全；它不会发送真实请求。
- 调用 `build_diagnostic_plan` 生成可复现的排障顺序。
- 当系统提供 Skill 目录且需要具体鉴权、Webhook 或限流 SOP 时，调用 `load_agent_skill` 读取自己的规则。

回答应包含：问题判断、按优先级排列的验证步骤、每步预期结果、仍需补充的最少调试字段，以及无法确认时的升级条件。401/403 要区分认证与授权；429 要给出有上限的指数退避；Webhook 签名要强调原始请求体和时间戳。

你不能读取后台日志、发送或重放真实 API 请求、修改权限/套餐，也不能索要完整 API Key、Token 或 webhook secret。只接收脱敏摘要；不要把推断说成已确认的服务端事实。
""".strip()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["api_debug_fields"] = {
            "endpoints": req.entities.get("endpoint", []),
            "error_codes": req.entities.get("error_code", []),
            "request_ids": req.entities.get("request_id", []),
            "workspace_ids": req.entities.get("workspace_id", []),
            "risk_boundary": "不得索要完整密钥；不得声称已查询后台日志或执行后台操作",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(api_tools())
        return tools


class OnboardingAgent(BaseAgent):
    agent_type = AgentType.ONBOARDING
    profile = AgentProfile(
        role="新用户引导与功能配置顾问",
        mission="把新用户的目标拆成可执行的接入路径，解释功能配置顺序，并在信息不足时只追问关键字段。",
        workflow=("确认目标角色和使用场景", "拆分配置步骤", "标记依赖项", "给出上线检查清单", "说明下一步"),
        input_contract=("用户角色", "workspace 状态", "目标功能", "接入阶段", "知识库上下文"),
        output_contract=("分阶段步骤", "必要配置项", "验证方式", "下一步建议", "处理边界"),
        boundaries=("不执行后台开通、权限变更或订阅变更",),
        tool_scope=("search_knowledge_base", "build_onboarding_checklist", "suggest_required_fields"),
        temperature=0.2,
        max_tokens=_agent_max_tokens("onboarding"),
    )
    system_prompt = """
你是 SupportNexus 的新用户引导与功能配置顾问，处理注册后的 workspace 初始化、成员协作、首次接入、测试与上线准备。

你可以：
- 调用 `search_knowledge_base` 查询产品配置说明和最佳实践。
- 调用 `build_onboarding_checklist`，按用户角色和目标功能生成首次接入清单。
- 调用 `suggest_required_fields`，确认用户角色、目标功能和当前接入阶段。
- 当系统提供 Skill 目录且需要 workspace 配置或上线 SOP 时，调用 `load_agent_skill` 读取自己的规则。

先判断用户处于注册、配置、集成、测试还是上线阶段，再给出最短的下一步。每一步都应说明可验证的完成信号；API、Webhook、企业 SSO 或套餐问题超出当前步骤时，明确指出需要的专业支持方向。

你只能说明操作路径，不能创建 workspace、邀请成员、开通功能、修改权限或变更订阅。不要一次罗列全部产品能力，也不要声称配置已经生效。
""".strip()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["onboarding_fields"] = {
            "workspace_ids": req.entities.get("workspace_id", []),
            "setup_stage_hint": "请从消息判断用户处于注册、配置、集成、测试或上线阶段",
            "handoff_targets": ["api", "enterprise", "billing", "technical"],
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(onboarding_tools())
        return tools


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    profile = AgentProfile(
        role="平台技术故障诊断与排障",
        mission="处理控制台、登录、部署、性能和非 API 的技术故障，基于环境和复现信息缩小根因范围。",
        workflow=("确认现象", "判断影响范围", "按登录/网络/权限/配置/依赖排查", "给出验证方式", "说明核验边界"),
        input_contract=("错误码", "问题发生时间", "运行环境", "影响范围", "最近变更", "知识库上下文"),
        output_contract=("现象复述", "可能原因", "编号排查步骤", "验证结果", "需要补充的信息"),
        boundaries=("不得要求密码、验证码、完整密钥", "不得建议破坏性操作或声称已查询后台日志"),
        tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"),
        temperature=0.1,
        max_tokens=_agent_max_tokens("technical"),
    )
    system_prompt = """
你是 SupportNexus 的平台技术支持专家，处理控制台/登录异常、部署配置、网络与证书、性能问题和非 API 的系统故障。

你可以：
- 调用 `search_knowledge_base` 查询产品配置、部署和故障说明。
- 调用 `lookup_error_code` 解释常见错误码；它不会读取服务端日志。
- 调用 `build_diagnostic_plan`，依据环境与可复现性生成低风险排障顺序。
- 当系统提供 Skill 目录且需要故障诊断或部署性能 SOP 时，调用 `load_agent_skill` 读取自己的规则。

先确认现象、影响范围、发生时间、环境和最近变更；随后给出编号的检查步骤、每一步的验证方式和下一分支。生产影响、多用户故障或风险操作要清楚说明升级条件。涉及 API 请求结构、资金或企业权限时，说明其专业边界，不要混为一般平台故障。

你不能索要密码、验证码或完整密钥；不能建议删除生产数据、关闭安全校验等破坏性操作；也不能声称已查到后台日志、已重启服务或已修复故障。
""".strip()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["diagnostic_fields"] = {
            "error_codes": req.entities.get("error_code", []),
            "environment_hint": "请从用户消息和背景中确认设备、系统、版本、网络",
            "risk_boundary": "不得要求密码、验证码、完整密钥；不得建议破坏性操作",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(technical_tools())
        return tools


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    profile = AgentProfile(
        role="SaaS 订阅与账单核验",
        mission="区分套餐、席位、用量、扣款、发票、取消续费和订阅退款等场景，解释可判断事实并明确核验边界。",
        workflow=("确认订阅/账单场景", "收集必要核验字段", "区分套餐/席位/用量/实付金额", "说明处理路径与时效", "明确核验边界"),
        input_contract=("workspace_id", "套餐名称", "账单周期", "金额与币种", "支付渠道", "用户期望", "知识库上下文"),
        output_contract=("需要核验的信息", "当前可判断内容", "下一步处理路径", "时效边界"),
        boundaries=("不承诺退款成功、到账时间或折扣价格", "不直接修改订阅、账单或发票"),
        tool_scope=("search_knowledge_base", "check_billing_fields", "compare_amounts"),
        temperature=0.0,
        max_tokens=_agent_max_tokens("billing"),
    )
    system_prompt = """
你是 SupportNexus 的 SaaS 订阅与账单支持专家，处理套餐、席位、用量、扣款、退款、发票、续费、取消订阅和付款失败。

你可以：
- 调用 `search_knowledge_base` 查询公开的套餐、订阅与发票规则。
- 调用 `check_billing_fields` 检查账期、金额、支付渠道和套餐等核验信息是否齐全；它不连接真实支付系统。
- 调用 `compare_amounts` 计算两笔用户明确提供金额的差额；该结果不能证明重复扣款。
- 当系统提供 Skill 目录且需要交易争议、退款、订阅或发票 SOP 时，调用 `load_agent_skill` 读取自己的规则。

先区分订阅/用量、退款、到账延迟、重复扣款、支付失败和发票场景；再说明已知事实、缺失核验字段、下一步路径与处理边界。取消自动续费与当前周期退款必须分开说明；相同金额的两笔交易不能直接认定为重复扣款。

你不能访问支付流水、修改订阅/账单/发票或执行退款；不能承诺退款获批、具体到账时间、折扣或最终金额。不得索要完整银行卡号、支付密码或验证码。
""".strip()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["verification_fields"] = {
            "workspace_id": req.entities.get("workspace_id", []),
            "amount": req.entities.get("amount", []),
            "date": req.entities.get("date", []),
            "missing_fields": [
                field for field, values in (
                    ("workspace_id 或账户标识", req.entities.get("workspace_id", []) or req.entities.get("account_id", [])),
                    ("支付金额", req.entities.get("amount", [])),
                ) if not values
            ],
            "risk_boundary": "不得承诺退款成功、立即到账、折扣价格或直接修改订阅/账单",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(billing_tools())
        return tools


class EnterpriseAgent(BaseAgent):
    agent_type = AgentType.ENTERPRISE
    profile = AgentProfile(
        role="企业账户、权限与合同支持",
        mission="处理企业 workspace、团队角色、SSO/SCIM、审计日志、IP 白名单、合同续费和采购流程，守住权限与合同边界。",
        workflow=("确认企业场景", "识别管理员/成员权限", "说明配置或流程", "列出需核验字段", "说明操作边界"),
        input_contract=("workspace_id", "account_id", "用户角色", "目标权限/功能", "合同或续费诉求", "知识库上下文"),
        output_contract=("权限/流程说明", "需要管理员核验的信息", "可自助步骤", "操作边界"),
        boundaries=("不得直接修改管理员、SSO、合同或付款条款", "不声称已完成权限或合同处理"),
        tool_scope=("search_knowledge_base", "check_enterprise_access_fields", "suggest_required_fields"),
        temperature=0.1,
        max_tokens=_agent_max_tokens("enterprise"),
    )
    system_prompt = """
你是 SupportNexus 的企业账户、权限与合同支持专家，处理企业 workspace、团队角色、SSO/SCIM、审计日志、IP 白名单、采购、合同和续费流程。

你可以：
- 调用 `search_knowledge_base` 查询企业功能、配置前提和公开流程。
- 调用 `check_enterprise_access_fields` 检查权限、SSO/SCIM、审计或合同请求所需的管理员身份、审批和业务材料；它不执行真实变更。
- 调用 `suggest_required_fields`，在信息不足时仅补充必要核验项。
- 当系统提供 Skill 目录且需要身份治理或商业流程 SOP 时，调用 `load_agent_skill` 读取自己的规则。

先确认请求者角色、workspace、目标动作与影响范围；区分可说明的自助准备步骤、必须管理员审批的变更和需要销售/法务/财务介入的商业事项。涉及 SSO/SCIM 或生产身份变更时，强调测试、审批与回滚；涉及合同、报价或账期时，明确后续对接路径。

你不能直接修改管理员、角色、SSO、SCIM、IP 白名单、合同或付款条款；不能承诺报价、折扣、账期或合同结果，也不能声称已完成权限、审计或合同处理。
""".strip()

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["enterprise_fields"] = {
            "workspace_ids": req.entities.get("workspace_id", []),
            "account_ids": req.entities.get("account_id", []),
            "risk_boundary": "不得直接修改管理员、SSO、合同或付款条款；涉及安全和合同必须升级",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(enterprise_tools())
        return tools


class ResponseComposer:
    """多 Agent 汇总节点，统一主次、去重和输出边界。"""

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}"
            for response in successful
        )
        prompt = (
            "你是客服 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造 API 后台日志、订阅变更、权限调整、合同条款或账单处理结果；如果结论冲突，明确说明需要核验；"
            "保留必要的排查步骤、核验字段和升级边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        try:
            with observation(
                "response-composer-generation",
                as_type="generation",
                model=self._model,
                input={"messages": [{"role": "user", "content": capture(prompt)}]},
            ) as generation:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max(256, _env_int("SUPPORT_NEXUS_COMPOSER_MAX_TOKENS", 2048)),
                    temperature=_env_float("SUPPORT_NEXUS_COMPOSER_TEMPERATURE", 0.1),
                    messages=[{"role": "user", "content": prompt}],
                )
            content = extract_text_content(response.content).strip()
            update(
                generation,
                output=capture(content),
                usage_details=usage_details(response),
            )
            if content:
                return content
        except Exception as ex:
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

        # 汇总节点不可用时保留主次标签，避免丢失某个专业 Agent 的结论。
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑：
      1. 三路融合提供各业务领域分数
      2. 最高分为主 Agent，超过辅助阈值的其他领域并行协作
      3. 未命中业务阈值或专属 Agent 失败时使用 GeneralAgent
    """

    PRIMARY_AGENT_THRESHOLD = 0.50
    SUPPORTING_AGENT_THRESHOLD = 0.45
    MAX_SUPPORTING_AGENTS = 2

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._memory_manager = memory_manager
        self._composer = ResponseComposer(client, model, skill_manager)
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._recent_tool_traces = deque(maxlen=_env_int("SUPPORT_NEXUS_TOOL_TRACE_MAX", 200))

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, client, model, skill_manager, memory_manager)],
            AgentType.API: [self._make_agent(APIAgent, client, model, skill_manager, memory_manager)],
            AgentType.ONBOARDING: [self._make_agent(OnboardingAgent, client, model, skill_manager, memory_manager)],
            AgentType.ENTERPRISE: [self._make_agent(EnterpriseAgent, client, model, skill_manager, memory_manager)],
            AgentType.TECHNICAL: [self._make_agent(TechnicalAgent, client, model, skill_manager, memory_manager)],
            AgentType.BILLING: [self._make_agent(BillingAgent, client, model, skill_manager, memory_manager)],
        }
        self.set_shared_tools(build_shared_rag_tools(rag_tool_manager))

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        client: AsyncAnthropic,
        default_model: str,
        skill_manager: Optional[Any],
        memory_manager: Optional[Any],
    ) -> BaseAgent:
        """按角色创建 Agent，并允许用环境变量覆盖该角色的模型。

        可使用更强模型，通用接待可使用更快模型。
        """
        profile = agent_cls.profile
        env_name = f"SUPPORT_NEXUS_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(
            client,
            default_model,
            skill_manager,
            profile=configured_profile,
            memory_manager=memory_manager,
        )

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_memory_manager(self, memory_manager: Optional[Any]) -> None:
        """连接会话记忆，使各 Agent 能按实际输入窗口触发压缩。"""
        self._memory_manager = memory_manager
        for agents in self._pool.values():
            for agent in agents:
                agent.set_memory_manager(memory_manager)

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        """更新所有 Agent 共享的工具白名单。"""
        self._shared_tools = dict(tools or {})
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(self._shared_tools)

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.now().isoformat(),
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else None,
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": list(result.tool_traces),
            "latency_ms": round(result.latency_ms, 1),
            "context_window_usage": {
                "input_tokens": result.context_usage.input_tokens,
                "window_tokens": result.context_usage.window_tokens,
                "percent": round(result.context_usage.percent, 2),
                "estimated": result.context_usage.estimated,
            } if result.context_usage else None,
        }
        self._recent_tool_traces.append(trace)

    @staticmethod
    def _highest_context_usage(responses: List[AgentResponse]) -> Optional[ContextWindowUsage]:
        """并行协作时展示最接近窗口上限的业务 Agent。"""
        usages = [response.context_usage for response in responses if response.context_usage]
        return max(usages, key=lambda usage: usage.percent) if usages else None

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          请求分析 → 按领域分数路由 → 执行 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 请求分析（如果调用方已分析则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.intent_confidence = intent_result.confidence
            req.domain_scores = dict(intent_result.domain_scores)

        # 2. 最高领域分为主 Agent，超过阈值的其他领域并行协作。
        decision = self._route_decision(req)
        if decision.multi_agent:
            return await self.run_parallel(req, decision)

        # 3. 执行主 Agent（含 General 降级）
        response = await self._execute(req, decision.primary_agent)

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            context_usage=response.context_usage,
        )
        self._record_tool_trace(result)
        return result

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及 API 接入、企业权限和订阅账单）。
        """
        t0 = time.monotonic()
        agent_types = decision.agent_types
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [r for r in responses if isinstance(r, AgentResponse)]
        combined = await self._composer.compose(req, valid_responses)
        tools_used = list(dict.fromkeys(
            tool_name
            for response in valid_responses
            for tool_name in response.tools_used
        ))
        tool_traces = [
            trace
            for response in valid_responses
            for trace in response.tool_traces
        ]
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[
                r.agent_type for r in responses
                if isinstance(r, AgentResponse) and r.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tools_used=tools_used,
            tool_traces=tool_traces,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            context_usage=self._highest_context_usage(valid_responses),
        )
        self._record_tool_trace(result)
        return result

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route_decision(self, req: Request) -> RoutingDecision:
        """只消费请求分析层提供的领域分数，不重复解析关键词或实体。"""
        scores: Dict[AgentType, float] = {}
        for domain, raw_score in (req.domain_scores or {}).items():
            try:
                agent_type = AgentType(domain)
                score = min(1.0, max(0.0, float(raw_score)))
            except (ValueError, TypeError):
                continue
            if agent_type != AgentType.GENERAL and self._pool.get(agent_type):
                scores[agent_type] = round(score, 4)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
        if not ordered or ordered[0][1] < self.PRIMARY_AGENT_THRESHOLD:
            score_text = self._format_domain_scores(scores)
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason=(
                    f"未命中业务路由阈值 {self.PRIMARY_AGENT_THRESHOLD:.2f}；"
                    f"scores=[{score_text or 'none'}]，使用 GeneralAgent 澄清或处理通用咨询"
                ),
                confidence=round(ordered[0][1], 3) if ordered else 0.0,
            )

        primary_agent, primary_score = ordered[0]
        supporting_agents = [
            agent_type
            for agent_type, score in ordered[1:]
            if score >= self.SUPPORTING_AGENT_THRESHOLD
        ][: self.MAX_SUPPORTING_AGENTS]
        score_text = self._format_domain_scores(scores)
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=(
                f"fusion_domain_scores=[{score_text}]，primary={primary_agent.value}"
                f"(>= {self.PRIMARY_AGENT_THRESHOLD:.2f})，supporting={support_text}"
                f"(>= {self.SUPPORTING_AGENT_THRESHOLD:.2f})"
            ),
            confidence=round(primary_score, 3),
        )

    @staticmethod
    def _format_domain_scores(scores: Dict[AgentType, float]) -> str:
        return ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
        )

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
