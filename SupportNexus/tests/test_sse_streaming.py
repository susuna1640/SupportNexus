import asyncio
import json
import sys
from types import SimpleNamespace

from agents.agent_orchestrator import AgentType, GeneralAgent, OrchestratorResult, Request


def _api_main():
    """Load the FastAPI module only in environments with server dependencies."""
    import pytest

    pytest.importorskip("fastapi")
    # The API module imports Uvicorn only for its ``__main__`` entrypoint. The
    # unit test exercises its FastAPI application object and needs no server.
    sys.modules.setdefault("uvicorn", SimpleNamespace())
    from api import main

    return main


class _Stream:
    def __init__(self, chunks, response):
        self._chunks = chunks
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    @property
    def text_stream(self):
        async def generate():
            for chunk in self._chunks:
                yield chunk

        return generate()

    async def get_final_message(self):
        return self._response


class _StreamClient:
    def __init__(self, chunks, response):
        self._chunks = chunks
        self._response = response
        self.messages = self

    def stream(self, **_kwargs):
        return _Stream(self._chunks, self._response)

    async def create(self, **_kwargs):
        raise AssertionError("streaming-capable client should not fall back to create")


def _request():
    return Request(
        message="如何排查 401？",
        user_id="u1",
        conv_id="c1",
        domain_scores={"api": 0.9},
    )


def test_agent_streams_model_text_deltas_and_keeps_complete_response():
    response = SimpleNamespace(content=[{"type": "text", "text": "请检查 Authorization。"}], stop_reason="end_turn")
    agent = GeneralAgent(_StreamClient(["请检查 ", "Authorization。"], response), "test-model")
    events = []

    async def publish(event, payload):
        events.append((event, payload))

    result = asyncio.run(agent.handle(_request(), event_callback=publish, stream_text=True))

    assert result.success is True
    assert result.content == "请检查 Authorization。"
    assert [payload["text"] for event, payload in events if event == "delta"] == ["请检查 ", "Authorization。"]
    assert [event for event, _payload in events] == ["agent_start", "delta", "delta", "agent_end"]


def test_agent_stream_emits_complete_text_when_provider_has_no_text_chunks():
    response = SimpleNamespace(content=[{"type": "text", "text": "完整回复。"}], stop_reason="end_turn")
    agent = GeneralAgent(_StreamClient([], response), "test-model")
    events = []

    async def publish(event, payload):
        events.append((event, payload))

    result = asyncio.run(agent.handle(_request(), event_callback=publish, stream_text=True))

    assert result.content == "完整回复。"
    assert [payload["text"] for event, payload in events if event == "delta"] == ["完整回复。"]


def test_sse_encoding_and_public_trace_exclude_tool_input_and_error_details():
    api_main = _api_main()
    frame = api_main._encode_sse("delta", {"text": "你好"})
    assert frame == 'event: delta\ndata: {"text":"你好"}\n\n'

    public_trace = api_main._public_tool_trace({
        "request_id": "r1",
        "tools_used": ["search_knowledge_base"],
        "tool_calls": [{
            "agent_type": "api",
            "tool_name": "search_knowledge_base",
            "input": {"query": "private user message"},
            "error": "internal detail",
            "success": True,
            "latency_ms": 12.3,
        }],
    })

    assert public_trace["tool_calls"] == [{
        "agent_type": "api",
        "tool_name": "search_knowledge_base",
        "success": True,
        "result_success": None,
        "latency_ms": 12.3,
        "cached": False,
        "reranked": False,
    }]
    assert "input" not in json.dumps(public_trace, ensure_ascii=False)
    assert "internal detail" not in json.dumps(public_trace, ensure_ascii=False)


def test_stream_endpoint_sends_progress_delta_and_sanitized_done_payload():
    api_main = _api_main()
    class Memory:
        async def get_context(self, *_args, **_kwargs):
            return SimpleNamespace(recent_messages=[], to_prompt_text=lambda **_kwargs: "")

        async def add_message(self, *_args, **_kwargs):
            return None

        async def record_completed_turn(self, *_args, **_kwargs):
            return False

    class Orchestrator:
        async def recognize_intent(self, *_args, **_kwargs):
            return SimpleNamespace(
                intent=SimpleNamespace(value="api_integration"),
                intent_group="api_integration",
                confidence=0.92,
                entities={},
                domain_scores={"api": 0.92},
                source_scores={"llm": 0.92},
            )

        async def run(self, _request, event_callback=None, stream_text=False):
            assert stream_text is True
            await event_callback("routing", {"primary_agent": "api", "message": "已完成 Agent 路由"})
            await event_callback("delta", {"text": "请检查 API Key。"})
            return OrchestratorResult(
                request_id="r1",
                response="请检查 API Key。",
                agent_type=AgentType.API,
                intent=SimpleNamespace(value="api_integration"),
                agent_types=[AgentType.API],
                primary_agent=AgentType.API,
                tools_used=["search_knowledge_base"],
            )

        def get_tool_trace(self, _request_id):
            return {
                "request_id": "r1",
                "tools_used": ["search_knowledge_base"],
                "tool_calls": [{
                    "tool_name": "search_knowledge_base",
                    "input": {"query": "must not leave server"},
                    "success": True,
                }],
            }

    async def collect():
        response = await api_main.chat_stream(api_main.ChatRequest(message="测试", user_id="u1"))
        return "".join([chunk async for chunk in response.body_iterator])

    previous_orchestrator, previous_memory = api_main._orchestrator, api_main._memory
    api_main._orchestrator, api_main._memory = Orchestrator(), Memory()
    try:
        body = asyncio.run(collect())
    finally:
        api_main._orchestrator, api_main._memory = previous_orchestrator, previous_memory

    assert "event: status" in body
    assert "event: routing" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert "must not leave server" not in body
    assert '"response":"请检查 API Key。"' in body
