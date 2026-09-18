"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果同时产出细粒度 intent 与业务领域分数。领域分数是 Agent 路由的
唯一输入：最高分选择主 Agent，其他超过阈值的领域选择辅助 Agent。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content
from core.observability import capture, observation, update, usage_details

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    API_INTEGRATION = "api_integration"      # API 对接问题
    FEATURE_INQUIRY = "feature_inquiry"      # 功能咨询
    ONBOARDING = "onboarding"                # 新用户引导
    ENTERPRISE_MGMT = "enterprise_mgmt"      # 企业账户管理
    SUBSCRIPTION = "subscription"            # 订阅管理
    ORDER_STATUS = "order_status"        # 订单状态
    LOGISTICS = "logistics"              # 物流配送
    REFUND = "refund"                    # 退款/退货
    INVOICE = "invoice"                  # 发票
    PAYMENT_ISSUE = "payment_issue"      # 支付/扣款异常
    ACCOUNT_SECURITY = "account_security" # 账户安全
    TECHNICAL_LOGIN = "technical_login"  # 登录认证故障
    TECHNICAL_CRASH = "technical_crash"  # 崩溃/错误码
    OTHER      = "other"


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    intent_group: str
    domain_scores: Dict[str, float]  # 融合后的业务领域分数，供主/辅 Agent 路由使用
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["这个怎么弄？", "在哪里查看 API 调用量？", "控制台入口在哪？", "在哪里查看基础信息？"],
    IntentCategory.COMPLAINT:  ["工单一直没人处理！", "你们这个接口文档太难用了！", "线上问题拖了很久还没解决！"],
    IntentCategory.REQUEST:    ["帮我导出本月用量明细", "我需要修改通知邮箱", "请协助开启测试环境"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.TECHNICAL:  ["控制台页面一直报500错误", "无法登录管理后台", "导入知识库任务一直失败"],
    IntentCategory.BILLING:    ["为什么这个月账单变高了？", "申请订阅退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["账号有问题", "修改账户邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
    IntentCategory.API_INTEGRATION: ["调用 /v1/chat 接口一直返回 401", "Webhook 签名校验失败", "SDK 接入时报 rate_limit_exceeded"],
    IntentCategory.FEATURE_INQUIRY: ["这个功能能不能接 Slack？", "企业版支持审计日志吗？", "是否支持自定义 Agent 数量？"],
    IntentCategory.ONBOARDING: ["我是新用户，怎么完成首次接入？", "如何创建 workspace 并邀请成员？", "上线前需要配置哪些功能？"],
    IntentCategory.ENTERPRISE_MGMT: ["企业合同续费流程是什么？", "如何配置 SSO 和 SCIM？", "团队成员权限怎么分配？"],
    IntentCategory.SUBSCRIPTION: ["专业版和企业版套餐差异是什么？", "怎么取消自动续费？", "本月订阅账单为什么超额？"],
    IntentCategory.REFUND: ["我想申请退款", "退款多久到账？", "如何退订并退款？"],
    IntentCategory.INVOICE: ["帮我开发票", "发票抬头填错了，可以重开吗？", "电子发票在哪里下载？"],
    IntentCategory.PAYMENT_ISSUE: ["为什么扣了两次款？", "支付失败但银行卡已经扣款", "扣款金额和账单不一致"],
    IntentCategory.ORDER_STATUS: ["我的订单什么时候到？", "订单状态在哪里看？", "订单号 12345 查一下进度"],
    IntentCategory.LOGISTICS: ["物流到哪里了？", "包裹还没送到", "配送超时怎么办？"],
    IntentCategory.ACCOUNT_SECURITY: ["API Key 疑似泄露了", "发现异常登录", "我要重置密码"],
    IntentCategory.TECHNICAL_LOGIN: ["登录一直报401", "验证码收不到", "无法登录账号"],
    IntentCategory.TECHNICAL_CRASH: ["应用一直崩溃", "页面报500错误", "系统闪退"],
    IntentCategory.OTHER: ["一直失败，是不是权限问题？", "今天天气怎么样？", "帮我写一首诗", "我忘了午饭吃什么了"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.API_INTEGRATION,
    IntentCategory.FEATURE_INQUIRY,
    IntentCategory.ONBOARDING,
    IntentCategory.ENTERPRISE_MGMT,
    IntentCategory.SUBSCRIPTION,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.TECHNICAL,
    IntentCategory.ACCOUNT,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.API_INTEGRATION: IntentCategory.API_INTEGRATION,
    IntentCategory.FEATURE_INQUIRY: IntentCategory.FEATURE_INQUIRY,
    IntentCategory.ONBOARDING: IntentCategory.ONBOARDING,
    IntentCategory.ENTERPRISE_MGMT: IntentCategory.ENTERPRISE_MGMT,
    IntentCategory.SUBSCRIPTION: IntentCategory.BILLING,
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
}

# 三路融合统一输出以下业务领域；general 是低置信度或非业务问题的兜底，
# 不参与领域分数竞争。
BUSINESS_DOMAINS = ("api", "onboarding", "enterprise", "technical", "billing")

_INTENT_TO_DOMAIN: Dict[IntentCategory, str] = {
    IntentCategory.API_INTEGRATION: "api",
    IntentCategory.FEATURE_INQUIRY: "onboarding",
    IntentCategory.ONBOARDING: "onboarding",
    IntentCategory.ENTERPRISE_MGMT: "enterprise",
    IntentCategory.ACCOUNT: "enterprise",
    IntentCategory.ACCOUNT_SECURITY: "enterprise",
    IntentCategory.TECHNICAL: "technical",
    IntentCategory.TECHNICAL_LOGIN: "technical",
    IntentCategory.TECHNICAL_CRASH: "technical",
    IntentCategory.BILLING: "billing",
    IntentCategory.SUBSCRIPTION: "billing",
    IntentCategory.REFUND: "billing",
    IntentCategory.INVOICE: "billing",
    IntentCategory.PAYMENT_ISSUE: "billing",
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        # Embedding 默认接阿里云百炼 / DashScope 的 OpenAI 兼容接口；未配置时
        # 保留本地字符 n-gram 向量兜底，避免意图识别因向量服务不可用而中断。
        self._embedding_enabled = True
        self._embedding_provider = os.getenv(
            "SUPPORT_NEXUS_INTENT_EMBEDDING_PROVIDER",
            "dashscope" if os.getenv("DASHSCOPE_API_KEY") else "local",
        ).strip().lower()
        self._embedding_model = os.getenv(
            "SUPPORT_NEXUS_INTENT_EMBEDDING_MODEL",
            "qwen3.7-text-embedding-flash",
        ).strip()
        self._embedding_batch_size = self._env_int("SUPPORT_NEXUS_INTENT_EMBEDDING_BATCH_SIZE", 25)
        self._embedding_dimensions = self._env_int("SUPPORT_NEXUS_INTENT_EMBEDDING_DIMENSIONS", 1024)
        self._embedding_timeout_s = self._env_float("SUPPORT_NEXUS_INTENT_EMBEDDING_TIMEOUT_S", 10.0)
        self._embedding_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        self._embedding_base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
        self._embedding_endpoint = os.getenv("DASHSCOPE_EMBEDDING_ENDPOINT", "").strip()
        self._embedding_workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
        self._embedding_region = os.getenv("DASHSCOPE_REGION", "cn-beijing").strip()
        self._embedding_strict = os.getenv(
            "SUPPORT_NEXUS_INTENT_EMBEDDING_STRICT",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._disable_llm_thinking = os.getenv(
            "SUPPORT_NEXUS_INTENT_LLM_DISABLE_THINKING",
            "true",
        ).strip().lower() not in {"0", "false", "no", "off"}

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self._usage_totals: Dict[str, int] = {
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "embedding_input_tokens": 0,
        }
        self._usage_available: Dict[str, bool] = {
            "llm": False,
            "embedding": False,
        }
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        domain_scores = self._fuse_domain_scores(llm, emb, pat)
        entities = self._extract_entities(message)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            intent_group=self._intent_group(intent),
            domain_scores=domain_scores,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    def usage_snapshot(self) -> Dict[str, Any]:
        """Return aggregate provider usage for evaluation and observability."""
        return {
            **self._usage_totals,
            "llm_usage_available": self._usage_available["llm"],
            "embedding_usage_available": self._usage_available["embedding"],
        }

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是 B2B SaaS 客服请求分析专家。根据示例判断用户的细粒度 intent，并评估每个业务领域是否需要处理，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如 API/SDK/Webhook 接入故障优先返回 api_integration，首次接入优先返回 onboarding，
企业 SSO/SCIM 或权限问题优先返回 enterprise_mgmt，订阅套餐问题优先返回 subscription，
登录故障优先返回 technical_login。

领域分数表示该领域是否应参与处理：单一问题通常只有一个高分；复合问题可有两个或更多高分。
只使用这些领域键：api、onboarding、enterprise、technical、billing。未涉及的领域设为 0。

模板示例：
{examples}

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "domain_scores": {{"api": <0-1>, "onboarding": <0-1>, "enterprise": <0-1>, "technical": <0-1>, "billing": <0-1>}}, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            with observation(
                "intent-classification-llm",
                as_type="generation",
                model=self.model,
                input={"messages": [{"role": "user", "content": capture(prompt)}]},
            ) as generation:
                request_kwargs = {
                    "model": self.model,
                    "max_tokens": self._env_int("SUPPORT_NEXUS_INTENT_LLM_MAX_TOKENS", 2048),
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self._disable_llm_thinking:
                    request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                resp = await self.client.messages.create(**request_kwargs)
            raw = extract_text_content(resp.content)
            token_usage = usage_details(resp)
            self._usage_totals["llm_input_tokens"] += token_usage.get("input_tokens", 0)
            self._usage_totals["llm_output_tokens"] += token_usage.get("output_tokens", 0)
            self._usage_available["llm"] = bool(token_usage)
            update(
                generation,
                output=capture(raw),
                usage_details=token_usage,
            )
            data = self._parse_llm_json(raw, resp)
            data = self._normalize_llm_intent(data)
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {
                "intent": IntentCategory.OTHER,
                "confidence": 0.0,
                "domain_scores": {},
                "reasoning": "LLM 失败",
                "error": str(ex),
                "failed": True,
            }

    def _parse_llm_json(self, raw: str, resp: Any) -> Dict[str, Any]:
        """Parse the first JSON object and expose provider-specific empty output."""
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            stop_reason = getattr(resp, "stop_reason", None)
            block_types = self._content_block_types(getattr(resp, "content", None))
            raise ValueError(
                "LLM 未返回可解析 JSON 正文；"
                f"stop_reason={stop_reason}; block_types={block_types}; raw_preview={raw[:120]!r}"
            )
        return json.loads(raw[start:end + 1])

    def _normalize_llm_intent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM outputs; intent grouping and allowed domains remain code-owned."""
        raw_intent = str(data.get("intent") or "").strip()

        try:
            intent = IntentCategory(raw_intent)
        except ValueError:
            intent = IntentCategory.OTHER

        data["intent"] = intent
        data["domain_scores"] = self._normalize_domain_scores(data.get("domain_scores"))
        if not any(data["domain_scores"].values()):
            domain = _INTENT_TO_DOMAIN.get(intent)
            if domain:
                data["domain_scores"][domain] = self._clamp_score(data.get("confidence", 0.0))
        # Ignore legacy/provider output: every path derives the group from intent.
        data.pop("intent_group", None)
        return data

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            intent_scores: Dict[IntentCategory, float] = {}
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                intent_scores[cat] = score
                if score > best_score:
                    best_score, best_cat = score, cat

            return {
                "intent": best_cat,
                "confidence": best_score,
                "domain_scores": self._domain_scores_from_intent_scores(intent_scores),
            }
        except Exception as ex:
            if self._embedding_strict:
                raise
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "domain_scores": {}}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        specific_patterns = {
            IntentCategory.GREETING: ["你好", "嗨", "hello", "hi", "在吗"],
            IntentCategory.API_INTEGRATION: [
                "api", "接口", "endpoint", "sdk", "webhook", "回调", "签名",
                "request_id", "请求头", "鉴权", "token", "api key", "限流",
                "rate limit", "rate_limit", "401", "403", "429",
            ],
            IntentCategory.ENTERPRISE_MGMT: [
                "企业账户", "企业版", "团队", "成员", "权限", "角色", "sso",
                "scim", "saml", "审计日志", "合同", "续费", "采购", "管理员",
            ],
            IntentCategory.ONBOARDING: [
                "新用户", "首次", "入门", "接入流程", "上线前", "创建 workspace",
                "创建工作区", "邀请成员", "配置", "初始化", "quickstart",
            ],
            IntentCategory.FEATURE_INQUIRY: [
                "支持", "功能", "能不能", "是否可以", "有没有", "可以接",
                "集成", "报表", "自定义", "audit log", "slack",
            ],
            IntentCategory.REFUND: ["退款", "退订", "退钱", "多久到账", "原路退回"],
            IntentCategory.INVOICE: ["发票", "开票", "抬头", "税号", "重开", "电子发票"],
            IntentCategory.PAYMENT_ISSUE: [
                "重复扣款", "扣了两次", "多扣", "支付失败", "银行卡已经扣款",
                "扣款金额", "payment failed",
            ],
            IntentCategory.SUBSCRIPTION: [
                "订阅", "套餐", "自动续费", "取消续费", "超额", "席位",
                "专业版", "套餐差异", "定价", "价格", "seat", "用量",
                "额度", "plan", "pricing",
            ],
            IntentCategory.ORDER_STATUS: ["订单", "订单号", "订单状态", "查一下进度"],
            IntentCategory.LOGISTICS: ["物流", "包裹", "配送", "快递", "送到", "超时"],
            IntentCategory.ACCOUNT_SECURITY: ["被盗", "异常登录", "重置密码", "两步验证", "安全"],
            IntentCategory.TECHNICAL_LOGIN: ["无法登录", "登录失败", "登录", "401", "验证码"],
            IntentCategory.TECHNICAL_CRASH: ["崩溃", "闪退", "500", "报错", "crash"],
            IntentCategory.OTHER: ["今天天气", "写一首诗", "午饭吃什么"],
        }
        generic_patterns = {
            IntentCategory.COMPLAINT:  ["投诉", "经理", "supervisor", "太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["账单", "订阅", "扣款", "发票", "billing", "invoice"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash", "超时", "失败"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }

        specific_scores = self._pattern_category_scores(msg, specific_patterns)
        generic_scores = self._pattern_category_scores(msg, generic_patterns)
        domain_scores = self._domain_scores_from_intent_scores({
            **generic_scores,
            **{
                category: max(score, generic_scores.get(category, 0.0))
                for category, score in specific_scores.items()
            },
        })

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score, "domain_scores": domain_scores}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score, "domain_scores": domain_scores}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """按细粒度 intent 加权投票；intent_group 在融合完成后统一映射。"""
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = float(result.get("confidence", 0.0) or 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        if best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS and pat_conf >= 0.5 and best_score < 0.8:
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    @staticmethod
    def _clamp_score(value: Any) -> float:
        """Normalize an untrusted score to the closed 0-1 range."""
        try:
            return round(min(1.0, max(0.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.0

    def _normalize_domain_scores(self, raw_scores: Any) -> Dict[str, float]:
        """Keep only router-owned domain names and valid numeric confidences."""
        values = raw_scores if isinstance(raw_scores, dict) else {}
        return {
            domain: self._clamp_score(values.get(domain, 0.0))
            for domain in BUSINESS_DOMAINS
        }

    def _domain_scores_from_intent_scores(
        self,
        intent_scores: Dict[IntentCategory, float],
    ) -> Dict[str, float]:
        """Aggregate fine-grained intent scores into the five routable domains."""
        scores = {domain: 0.0 for domain in BUSINESS_DOMAINS}
        for intent, value in intent_scores.items():
            domain = _INTENT_TO_DOMAIN.get(intent)
            if domain:
                scores[domain] = max(scores[domain], self._clamp_score(value))
        return scores

    def _fuse_domain_scores(self, llm: Dict, emb: Dict, pat: Dict) -> Dict[str, float]:
        """Fuse the three recognizers once; routing consumes this result directly."""
        if self._embedding_enabled:
            weighted_sources = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weighted_sources = [(llm, 0.85), (pat, 0.15)]

        active_sources = [
            (result, weight)
            for result, weight in weighted_sources
            if not result.get("failed")
        ]
        total_weight = sum(weight for _, weight in active_sources)
        if not total_weight:
            return {domain: 0.0 for domain in BUSINESS_DOMAINS}

        fused = {domain: 0.0 for domain in BUSINESS_DOMAINS}
        for result, weight in active_sources:
            raw_scores = result.get("domain_scores")
            scores = self._normalize_domain_scores(raw_scores)
            if not any(scores.values()):
                domain = _INTENT_TO_DOMAIN.get(result.get("intent"))
                if domain:
                    scores[domain] = self._clamp_score(result.get("confidence", 0.0))
            for domain, score in scores.items():
                fused[domain] += score * weight / total_weight
        return {domain: round(score, 4) for domain, score in fused.items()}

    @staticmethod
    def _pattern_category_scores(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> Dict[IntentCategory, float]:
        """Return every matched intent score so a composite request retains all domains."""
        scores: Dict[IntentCategory, float] = {}
        for category, keywords in patterns.items():
            hits = sum(1 for keyword in keywords if keyword in message)
            if hits:
                scores[category] = min(1.0, 0.5 + 0.25 * (hits - 1))
        return scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        return {
            "order_id": self._unique(re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)),
            "request_id": self._unique(re.findall(r"(?:request[_ -]?id|请求[ _-]?id)\s*[:：#]?\s*([A-Za-z0-9_-]{6,64})", message, re.I)),
            "workspace_id": self._unique(re.findall(r"(?:workspace[_ -]?id|工作区[ _-]?id)\s*[:：#]?\s*([A-Za-z0-9_-]{4,64})", message, re.I)),
            "account_id": self._unique(re.findall(r"(?:account[_ -]?id|账号[ _-]?id|账户[ _-]?id)\s*[:：#]?\s*([A-Za-z0-9_-]{4,64})", message, re.I)),
            "endpoint": self._unique(re.findall(r"(?:(?:GET|POST|PUT|PATCH|DELETE)\s+)?(/v\d+/[A-Za-z0-9_./{}-]+)", message, re.I)),
            "product": [],
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", message, re.I)),
            "error_code": self._unique(
                re.findall(r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b", message, re.I)
                + re.findall(r"\b([45]\d{2})\b", message)
                + re.findall(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\b", message)
            ),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = await self._embed_texts(all_texts)
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """生成单条文本向量。"""
        return (await self._embed_texts([text]))[0]

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量。

        优先使用阿里云百炼 / DashScope 的 OpenAI 兼容 Embeddings API；
        未配置或调用失败时，回退到本地字符 n-gram 哈希向量。
        """
        clean_texts = [self._clean_text(text) for text in texts]
        if self._should_use_dashscope_embedding():
            try:
                return await self._dashscope_embeddings(clean_texts)
            except Exception as ex:
                if self._embedding_strict:
                    raise
                logger.warning(f"DashScope Embedding 失败，使用本地向量兜底: {ex}")

        return [self._local_embedding(text) for text in clean_texts]

    def _should_use_dashscope_embedding(self) -> bool:
        return (
            self._embedding_provider in {"auto", "dashscope", "aliyun", "bailian"}
            and bool(self._embedding_api_key)
            and bool(self._embedding_model)
        )

    async def _dashscope_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用阿里云百炼 / DashScope OpenAI 兼容 Embeddings API。"""
        batch_size = max(1, min(self._embedding_batch_size, 25))
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(await self._dashscope_embeddings_batch(texts[start:start + batch_size]))
        return vectors

    async def _dashscope_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """调用一次 DashScope Embeddings API，单批最多 25 条。"""
        url = self._dashscope_embeddings_url()
        payload: Dict[str, Any] = {
            "model": self._embedding_model,
            "input": texts,
        }
        if self._embedding_dimensions > 0:
            payload["dimensions"] = self._embedding_dimensions

        with observation(
            "intent-embedding-dashscope",
            as_type="embedding",
            model=self._embedding_model,
            input={
                "text_count": len(texts),
                "dimensions": self._embedding_dimensions,
            },
        ) as embedding_observation:
            data = await self._post_embedding_json(url, payload)

        usage = data.get("usage") or {}
        token_count = usage.get("prompt_tokens", usage.get("input_tokens", usage.get("total_tokens")))
        try:
            if token_count is not None:
                self._usage_totals["embedding_input_tokens"] += int(token_count)
                self._usage_available["embedding"] = True
        except (TypeError, ValueError):
            logger.debug("Embedding usage 无法解析: %r", usage)

        vectors: List[List[float]] = []
        for index, item in enumerate(data.get("data", [])):
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise ValueError(f"DashScope 返回缺少 embedding: index={index}")
            vectors.append([float(value) for value in vector])

        if len(vectors) != len(texts):
            raise ValueError(f"DashScope 返回向量数量不匹配: expected={len(texts)} actual={len(vectors)}")
        update(
            embedding_observation,
            output={
                "vector_count": len(vectors),
                "vector_dim": len(vectors[0]) if vectors else 0,
            },
        )
        return vectors

    def _dashscope_embeddings_url(self) -> str:
        if self._embedding_endpoint:
            return self._embedding_endpoint

        if self._embedding_base_url:
            base = self._embedding_base_url.rstrip("/")
            if base.endswith("/embeddings"):
                return base
            return f"{base}/embeddings"

        if self._embedding_workspace_id:
            return (
                f"https://{self._embedding_workspace_id}.{self._embedding_region}.maas.aliyuncs.com"
                "/compatible-mode/v1/embeddings"
            )

        return "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

    async def _post_embedding_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Post JSON with httpx when available, otherwise use stdlib urllib."""
        headers = {
            "Authorization": f"Bearer {self._embedding_api_key}",
            "Content-Type": "application/json",
        }
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._embedding_timeout_s) as client:
                response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except ModuleNotFoundError:
            return await asyncio.to_thread(
                self._post_embedding_json_urllib,
                url,
                headers,
                payload,
            )

    def _post_embedding_json_urllib(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        import urllib.error
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._embedding_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            detail = ex.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope HTTP {ex.code}: {detail}") from ex

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)[:200]}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @staticmethod
    def _content_block_types(content: Any) -> List[str]:
        types = []
        for block in content or []:
            if isinstance(block, dict):
                types.append(str(block.get("type", type(block).__name__)))
            else:
                types.append(str(getattr(block, "type", type(block).__name__)))
        return types

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
            return default

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
