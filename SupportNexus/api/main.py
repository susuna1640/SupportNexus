"""
SupportNexus 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   SupportNexus  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request, build_shared_rag_tools
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from core.observability import shutdown as shutdown_langfuse

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("SUPPORT_NEXUS_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("SUPPORT_NEXUS_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆 + 文件式长期用户记忆）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        user_memory_dir=os.getenv("SUPPORT_NEXUS_USER_MEMORY_DIR") or None,
    )
    _orchestrator.set_memory_manager(_memory)

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager()
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或补充更多上下文后再次查询。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 单路向量检索，用作 RAG 基线）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=False,
        fallback=knowledge_fallback,
    ))
    _tool_manager.register(Tool(
        name="knowledge_hybrid_search",
        description="搜索知识库（ChromaDB 稠密检索 + BM25 + RRF 融合）",
        handler=kb.hybrid_search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))
    if _orchestrator is not None:
        _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("SupportNexus 已就绪")
    yield

    await _monitor.stop()
    if _memory is not None:
        await _memory.close()
    shutdown_langfuse()
    logger.info("SupportNexus 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SupportNexus 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    request_id:  str = ""
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    latency_ms:  float
    knowledge_used: bool = False
    context_input_tokens: Optional[int] = None
    context_window_tokens: Optional[int] = None
    context_window_usage_percent: Optional[float] = None
    context_window_usage_estimated: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    domain_scores: Dict[str, float] = Field(default_factory=dict)


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole
    from core.observability import capture, observation, trace_attributes, update

    conv_id = req.conv_id or str(uuid.uuid4())

    with observation(
        "support-nexus-chat-request",
        as_type="agent",
        input={
            "message": capture(req.message),
            "conversation_id": conv_id,
        },
        metadata={"feature": "customer_support", "application": "support-nexus"},
    ) as root_observation:
        with trace_attributes(
            user_id=str(req.user_id)[:200],
            session_id=str(conv_id)[:200],
            tags=["support-nexus", "customer-support"],
            version="2.0.0",
            metadata={"feature": "customer_support"},
        ):
            # 1. 读取记忆上下文
            with observation(
                "conversation-memory-read",
                as_type="span",
                input={"user_id": str(req.user_id)[:200], "conversation_id": conv_id},
            ) as memory_observation:
                mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
                update(
                    memory_observation,
                    output={
                        "recent_message_count": len(mem_ctx.recent_messages),
                        "has_user_memories": bool(mem_ctx.user_memories),
                        "has_semantic_memory": bool(mem_ctx.relevant_history),
                    },
                )

            # 2. 构建编排请求（含对话历史，用于意图识别上下文）
            history = [
                {"role": m.role.value, "content": m.content}
                for m in mem_ctx.recent_messages[-5:]
            ] if mem_ctx.recent_messages else None

            with observation(
                "intent-recognition",
                as_type="chain",
                input={"message": capture(req.message), "history_turns": len(history or [])},
            ) as intent_observation:
                intent_result = await _orchestrator.recognize_intent(req.message, history=history)
                update(
                    intent_observation,
                    output={
                        "intent": intent_result.intent.value,
                        "intent_group": intent_result.intent_group,
                        "confidence": intent_result.confidence,
                        "entities": intent_result.entities,
                        "domain_scores": intent_result.domain_scores,
                        "source_scores": intent_result.source_scores,
                    },
                )
            full_context = mem_ctx.to_prompt_text(
                max_message_chars=_memory.CONTEXT_MESSAGE_MAX_CHARS,
            )

            orch_req = OrcReq(
                message=req.message,
                user_id=req.user_id,
                conv_id=conv_id,
                context=full_context,
                history=history,
                entities=intent_result.entities,
                domain_scores=intent_result.domain_scores,
                intent=intent_result.intent,
                intent_group=intent_result.intent_group,
                intent_confidence=intent_result.confidence,
            )

            # 3. 执行
            with observation(
                "agent-orchestration",
                as_type="agent",
                input={
                    "message": capture(req.message),
                    "intent": intent_result.intent.value,
                    "intent_confidence": intent_result.confidence,
                },
            ) as orchestration_observation:
                result = await _orchestrator.run(orch_req)
                update(
                    orchestration_observation,
                    output={
                        "response": capture(result.response),
                        "agent_type": result.agent_type.value,
                        "agent_types": [agent.value for agent in result.agent_types],
                        "tools_used": result.tools_used,
                        "routing_reason": result.routing_reason,
                        "context_window_usage_percent": round(result.context_usage.percent, 2)
                        if result.context_usage else None,
                        "context_window_usage_estimated": bool(
                            result.context_usage and result.context_usage.estimated
                        ),
                    },
                )

            # 4. 写入记忆
            with observation("conversation-memory-write", as_type="span") as memory_observation:
                await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)
                update(memory_observation, output={"messages_written": 2})

            # 5. 每五组完整问答异步审阅一次长期用户记忆，不阻塞响应。
            memory_review_due = await _memory.record_completed_turn(req.user_id, conv_id)
            if memory_review_due:
                asyncio.create_task(_memory.update_user_memories(req.user_id, conv_id))

            response = ChatResponse(
                conv_id=conv_id,
                request_id=result.request_id,
                response=result.response,
                intent=result.intent.value if result.intent else "other",
                intent_group=intent_result.intent_group,
                agent_type=result.agent_type.value,
                agent_types=[agent_type.value for agent_type in result.agent_types],
                primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
                supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
                tools_used=result.tools_used,
                routing_reason=result.routing_reason,
                routing_confidence=result.routing_confidence,
                latency_ms=round(result.latency_ms, 1),
                knowledge_used="search_knowledge_base" in result.tools_used,
                context_input_tokens=result.context_usage.input_tokens if result.context_usage else None,
                context_window_tokens=result.context_usage.window_tokens if result.context_usage else None,
                context_window_usage_percent=round(result.context_usage.percent, 2)
                if result.context_usage else None,
                context_window_usage_estimated=bool(
                    result.context_usage and result.context_usage.estimated
                ),
                entities=intent_result.entities,
                intent_confidence=round(intent_result.confidence, 4),
                intent_source_scores=intent_result.source_scores,
                domain_scores=intent_result.domain_scores,
            )
            update(root_observation, output=capture(response.model_dump()))
            return response


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5, use_cache: bool = True):
    """
    演示检索优化链路：Agent query → Hybrid 召回 → Qwen3-Rerank → Top-K。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_optimized(
        "knowledge_hybrid_search",
        query,
        top_k=top_k,
        use_cache=use_cache,
    )
    return {
        "query": query,
        "results": result.data,
        "reranked": result.reranked,
        "cached": result.cached,
        "cache_enabled": use_cache,
        "strategy": "hybrid_rrf_qwen3_rerank",
    }


@app.post("/search/baseline")
async def search_baseline(query: str, top_k: int = 5):
    """RAG 基线：原始 query -> 单路向量检索 -> Top-K。"""
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_baseline("knowledge_search", query, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": result.data,
        "success": result.success,
        "reranked": False,
        "strategy": "single_vector_baseline",
        "error": result.error,
    }


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async([{"title": d.title, "content": d.content} for d in body.documents])
    _invalidate_knowledge_search_cache()
    total = await kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = await kb.add_documents_async(docs)
    _invalidate_knowledge_search_cache()
    total = await kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": await kb.doc_count_async()}


def _knowledge_base():
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    return tool.handler.__self__


def _invalidate_knowledge_search_cache() -> int:
    if _tool_manager is None:
        return 0
    return (
        _tool_manager.clear_cache("knowledge_search")
        + _tool_manager.clear_cache("knowledge_hybrid_search")
    )


@app.get("/knowledge/chunks", tags=["知识库"])
async def list_knowledge_chunks(limit: int = 100, offset: int = 0):
    """列出当前入库文本块及其切片元数据。"""
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit 必须在 1 到 500 之间")
    kb = _knowledge_base()
    chunks = await kb.list_chunks_async(limit=limit, offset=offset)
    return {
        "total_chunks": await kb.doc_count_async(),
        "offset": offset,
        "chunks": chunks,
    }


@app.delete("/knowledge/chunks", tags=["知识库"])
async def delete_knowledge_chunks(title: str):
    """按 title 删除对应原文的所有文本块。"""
    kb = _knowledge_base()
    deleted = await kb.delete_chunks_by_title_async(title)
    return {
        "title": title,
        "deleted_chunks": deleted,
        "cleared_cache_entries": _invalidate_knowledge_search_cache(),
        "total_chunks": await kb.doc_count_async(),
    }


@app.delete("/knowledge/chunks/all", tags=["知识库"])
async def clear_knowledge_chunks(confirm: bool = False):
    """清空全部知识库 chunk；必须显式传入 confirm=true。"""
    if not confirm:
        raise HTTPException(400, "清空知识库需要 confirm=true")
    kb = _knowledge_base()
    deleted = await kb.clear_chunks_async()
    return {
        "deleted_chunks": deleted,
        "cleared_cache_entries": _invalidate_knowledge_search_cache(),
        "total_chunks": await kb.doc_count_async(),
    }


@app.post("/knowledge/seed-defaults", tags=["知识库"])
async def seed_default_knowledge():
    """向空知识库显式导入项目内置的 B2B SaaS 样例文档。"""
    kb = _knowledge_base()
    try:
        seeded = await kb.seed_default_documents_async()
    except ValueError as ex:
        raise HTTPException(409, str(ex)) from ex
    return {
        "seeded_chunks": seeded,
        "cleared_cache_entries": _invalidate_knowledge_search_cache(),
        "total_chunks": await kb.doc_count_async(),
    }


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("SupportNexus CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("SUPPORT_NEXUS_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("SUPPORT_NEXUS_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        user_memory_dir=os.getenv("SUPPORT_NEXUS_USER_MEMORY_DIR") or None,
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    orch.set_memory_manager(mem)

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(
            message=msg,
            user_id=user_id,
            conv_id=conv_id,
            context=ctx.to_prompt_text(max_message_chars=mem.CONTEXT_MESSAGE_MAX_CHARS),
            history=history,
        )
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)
        if await mem.record_completed_turn(user_id, conv_id):
            asyncio.create_task(mem.update_user_memories(user_id, conv_id))

        print(f"\nSupportNexus [{result.agent_type.value}]: {result.response}\n")

    await mem.close()
    from core.observability import shutdown as shutdown_langfuse
    shutdown_langfuse()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
