import asyncio

import httpx
import pytest

from mcp.reranker import Qwen3Reranker


def test_qwen3_reranker_uses_document_indices_and_preserves_retrieval_score():
    reranker = Qwen3Reranker(
        api_key="test",
        workspace_id="workspace-id",
        model="qwen3-rerank",
    )
    payloads = []

    async def fake_post_json(payload):
        payloads.append(payload)
        return {
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.31},
            ]
        }

    reranker._post_json = fake_post_json
    items = [
        {"id": "a", "title": "退款流程", "content": "退款需要审核", "score": 0.021},
        {"id": "b", "title": "到账时间", "content": "退款三个工作日到账", "score": 0.031},
        {"id": "c", "title": "账户安全", "content": "请开启双重验证", "score": 0.011},
    ]

    result = asyncio.run(reranker.rerank("退款什么时候到账", items, top_k=2))

    assert result.applied is True
    assert [item["id"] for item in result.items] == ["b", "a"]
    assert result.items[0]["score"] == 0.92
    assert result.items[0]["retrieval_score"] == 0.031
    assert payloads[0]["query"] == "退款什么时候到账"
    assert payloads[0]["top_n"] == 2
    assert payloads[0]["documents"][0].startswith("标题：退款流程")


def test_qwen3_reranker_exposes_dashscope_error_body(monkeypatch):
    reranker = Qwen3Reranker(api_key="test", workspace_id="workspace-id")

    class FakeResponse:
        status_code = 400
        text = '{"code":"InvalidParameter","message":"model is unavailable"}'

        def raise_for_status(self):
            request = httpx.Request("POST", "https://example.test/reranks")
            raise httpx.HTTPStatusError("400 Bad Request", request=request, response=self)

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    with pytest.raises(RuntimeError, match="InvalidParameter"):
        asyncio.run(reranker._post_json({"model": "qwen3-rerank"}))
