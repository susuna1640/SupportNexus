import asyncio

from mcp.tool_manager import MCPToolManager, Tool
from mcp.reranker import RerankResult


def test_baseline_uses_one_original_query_without_cache_or_rerank():
    calls = []

    async def handler(params, context):
        calls.append((params, context))
        return [{"title": "退款政策", "content": "退款将在原路退回。"}]

    manager = MCPToolManager.__new__(MCPToolManager)
    manager._tools = {}
    manager._cache = {}
    manager.register(Tool(
        name="knowledge_search",
        description="test search tool",
        handler=handler,
        schema={"type": "object", "required": ["query", "top_k"], "properties": {}},
        cache_ttl=60,
        supports_rerank=True,
    ))

    result = asyncio.run(manager.search_baseline("knowledge_search", "退款什么时候到账", top_k=5))

    assert result.success is True
    assert result.reranked is False
    assert result.cached is False
    assert calls == [({"query": "退款什么时候到账", "top_k": 5}, None)]
    assert manager._cache == {}


def test_optimized_search_uses_agent_query_once_then_qwen_reranks():
    tool_calls = []

    class FakeReranker:
        model = "qwen3-rerank"

        def __init__(self):
            self.calls = []

        async def rerank(self, query, items, top_k):
            self.calls.append((query, items, top_k))
            return RerankResult(items=list(reversed(items))[:top_k], applied=True, model=self.model)

    async def handler(params, context):
        tool_calls.append(params)
        return [
            {"id": "shared", "content": "退款将在原路到账", "score": 0.03},
            {"id": "billing-policy", "content": "退款规则说明", "score": 0.01},
            {"id": "unrelated", "content": "管理员可配置成员角色", "score": 0.005},
        ]

    manager = MCPToolManager.__new__(MCPToolManager)
    manager._tools = {}
    manager._cache = {}
    manager._recall_top_k = 3
    manager._rerank_candidate_k = 3
    manager._reranker = FakeReranker()
    manager.register(Tool(
        name="knowledge_hybrid_search",
        description="test hybrid search tool",
        handler=handler,
        schema={"type": "object", "required": ["query", "top_k"], "properties": {}},
        supports_rerank=True,
    ))

    result = asyncio.run(manager.search_optimized("knowledge_hybrid_search", "退款到账", top_k=2))

    assert result.success is True
    assert result.reranked is True
    assert tool_calls == [{"query": "退款到账", "top_k": 3}]
    assert all(call["top_k"] == 3 for call in tool_calls)
    _, candidates, top_k = manager._reranker.calls[0]
    assert [candidate["id"] for candidate in candidates] == ["shared", "billing-policy", "unrelated"]
    assert top_k == 2


def test_optimized_search_can_disable_retrieval_cache_for_evaluation():
    async def handler(params, context):
        return [{"id": "one", "content": "证据"}, {"id": "two", "content": "辅助证据"}]

    manager = MCPToolManager.__new__(MCPToolManager)
    manager._tools = {}
    manager._cache = {}
    manager._recall_top_k = 2
    manager._rerank_candidate_k = 2
    manager._reranker = None
    manager.register(Tool(
        name="knowledge_hybrid_search",
        description="test hybrid search tool",
        handler=handler,
        schema={"type": "object", "required": ["query", "top_k"], "properties": {}},
        cache_ttl=60,
        supports_rerank=True,
    ))

    result = asyncio.run(
        manager.search_optimized("knowledge_hybrid_search", "评测 query", top_k=2, use_cache=False)
    )

    assert result.success is True
    assert result.cached is False
    assert manager._cache == {}


def test_clear_cache_removes_only_the_requested_tool_entries():
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._cache = {
        "knowledge_search:one": ([], 0, False),
        "knowledge_hybrid_search:two": ([], 0, False),
        "other_tool:three": ([], 0, False),
    }

    removed = manager.clear_cache("knowledge_search")

    assert removed == 1
    assert set(manager._cache) == {"knowledge_hybrid_search:two", "other_tool:three"}
