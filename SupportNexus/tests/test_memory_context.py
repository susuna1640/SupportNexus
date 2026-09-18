import asyncio
import json
from datetime import datetime, timedelta

import pytest


pytest.importorskip("chromadb")
pytest.importorskip("redis.asyncio")

from memory.conversation_memory import MemoryContext, MemoryManager, Message, MsgRole


def test_memory_context_uses_summary_recent_user_memories_then_cross_session_history():
    original_long_message = "A" * 80
    context = MemoryContext(
        summary="用户正在排查 API 401，尚未提供 request_id。",
        recent_messages=[
            Message(role=MsgRole.USER, content=original_long_message),
            Message(role=MsgRole.ASSISTANT, content="请提供 request_id。"),
        ],
        user_memories=["用户偏好使用中文交流", "用户使用企业版套餐"],
        relevant_history=["另一会话曾确认 workspace 为 ws_prod_1234。"],
    )

    prompt = context.to_prompt_text(max_message_chars=40)

    assert prompt.index("[会话摘要]") < prompt.index("[最近对话]")
    assert prompt.index("[最近对话]") < prompt.index("[长期用户记忆]")
    assert prompt.index("[长期用户记忆]") < prompt.index("[跨会话相关历史]")
    assert "[单条消息已截断]" in prompt
    assert context.recent_messages[0].content == original_long_message


def test_episodic_search_includes_closed_history_from_the_same_conversation_id():
    manager = MemoryManager.__new__(MemoryManager)

    async def fake_query(query_text, n_results, where):
        assert query_text == "权限问题"
        assert where == {"user_id": "u1"}
        assert n_results == manager.HISTORY_QUERY_CANDIDATES
        return {
            "documents": [["当前会话快照", "另一会话的权限核验记录", "另一会话的权限核验记录"]],
            "metadatas": [[
                {"conv_id": "active", "kind": "closed_session_summary"},
                {"conv_id": "older"},
                {"conv_id": "older"},
            ]],
        }

    manager._query_episodic = fake_query

    history = asyncio.run(manager._search_episodic("u1", "active", "权限问题"))

    assert history == ["当前会话快照", "另一会话的权限核验记录"]


def test_idle_session_is_archived_before_context_is_loaded():
    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.lists = {}
            self.expirations = []
            self.deleted = []

        async def get(self, key):
            return self.values.get(key)

        async def set(self, key, value, nx=False):
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True

        async def setex(self, key, seconds, value):
            self.values[key] = value
            self.expirations.append((key, seconds))

        async def expire(self, key, seconds):
            self.expirations.append((key, seconds))

        async def lrange(self, key, _start, _end):
            return list(self.lists.get(key, []))

        async def lpush(self, key, value):
            self.lists.setdefault(key, []).insert(0, value)

        async def delete(self, *keys):
            self.deleted.extend(keys)
            for key in keys:
                self.values.pop(key, None)
                self.lists.pop(key, None)

    now = datetime(2026, 9, 17, 12, 0, 0)
    started_at = now - timedelta(days=1, minutes=5)
    redis = FakeRedis()
    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = redis
    manager._session_locks = {}
    manager._now = lambda: now

    user_id, conv_id = "u1", "c1"
    wm_key = manager._wm_key(user_id, conv_id)
    first = Message(role=MsgRole.USER, content="请继续排查 API 401", timestamp=started_at)
    second = Message(
        role=MsgRole.ASSISTANT,
        content="需要提供 request_id。",
        timestamp=started_at + timedelta(minutes=1),
    )
    # Redis 工作列表以最新消息在前的顺序保存。
    redis.lists[wm_key] = [
        json.dumps({"role": second.role.value, "content": second.content, "ts": second.timestamp.isoformat(), "metadata": {}}),
        json.dumps({"role": first.role.value, "content": first.content, "ts": first.timestamp.isoformat(), "metadata": {}}),
    ]
    redis.values[manager._summary_key(user_id, conv_id)] = "此前已确认使用生产环境。"
    redis.values[manager._last_activity_key(user_id, conv_id)] = second.timestamp.isoformat()
    redis.values[manager._session_started_key(user_id, conv_id)] = first.timestamp.isoformat()

    archived = []

    async def summarize(old_summary, messages):
        assert old_summary == "此前已确认使用生产环境。"
        assert [message.content for message in messages] == [first.content, second.content]
        return "API 401：生产环境，仍需 request_id。"

    async def store_closed(*args):
        archived.append(args)
        return True

    async def user_memories(_user_id):
        return [("m_001", "用户偏好中文技术说明")]

    async def search_history(_user_id, _conv_id, query):
        assert query == "请继续排查 API 401"
        return ["API 401：生产环境，仍需 request_id。"]

    manager._summarize_session_end = summarize
    manager._store_closed_session_summary = store_closed
    manager._get_user_memory_entries = user_memories
    manager._search_episodic = search_history

    context = asyncio.run(manager.get_context(user_id, conv_id, query="请继续排查 API 401"))

    assert len(archived) == 1
    assert archived[0][0:2] == (user_id, conv_id)
    assert context.summary == ""
    assert context.recent_messages == []
    assert context.user_memories == ["用户偏好中文技术说明"]
    assert context.relevant_history == ["API 401：生产环境，仍需 request_id。"]
    assert wm_key in redis.deleted
    assert manager._summary_key(user_id, conv_id) in redis.deleted
    assert manager._last_activity_key(user_id, conv_id) in redis.deleted


def test_normal_compaction_updates_redis_without_writing_episodic_memory():
    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.lists = {}

        async def get(self, key):
            return self.values.get(key)

        async def setex(self, key, _seconds, value):
            self.values[key] = value

        async def lrange(self, key, _start, _end):
            return list(self.lists.get(key, []))

        async def lpush(self, key, value):
            self.lists.setdefault(key, []).insert(0, value)

        async def delete(self, *keys):
            for key in keys:
                self.values.pop(key, None)
                self.lists.pop(key, None)

        async def expire(self, _key, _seconds):
            return True

    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = FakeRedis()
    manager._session_locks = {}
    manager._model = "test-model"

    user_id, conv_id = "u1", "c1"
    messages = [
        Message(role=MsgRole.USER, content="第一条"),
        Message(role=MsgRole.ASSISTANT, content="第二条"),
        Message(role=MsgRole.USER, content="第三条"),
        Message(role=MsgRole.ASSISTANT, content="第四条"),
        Message(role=MsgRole.USER, content="第五条"),
    ]
    manager._redis.lists[manager._wm_key(user_id, conv_id)] = [
        json.dumps({"role": message.role.value, "content": message.content, "ts": message.timestamp.isoformat(), "metadata": {}})
        for message in reversed(messages)
    ]

    compacted = asyncio.run(manager.compact_session(user_id, conv_id, keep_recent_messages=4))

    assert compacted is True
    assert manager._redis.values[manager._summary_key(user_id, conv_id)]
    assert len(manager._redis.lists[manager._wm_key(user_id, conv_id)]) == 4


def test_user_memory_operations_are_prefix_free_and_ignore_no_change():
    manager = MemoryManager.__new__(MemoryManager)
    existing = [
        ("m_001", "用户偏好先讨论架构，再决定是否修改代码"),
        ("m_002", "用户使用 FastAPI、Redis 和 ChromaDB"),
    ]

    updated = manager._apply_user_memory_operations(
        existing,
        """用户希望回复中展示上下文窗口占用比例
m_002 § 用户使用 FastAPI、异步 Redis 和 ChromaDB
m_001 §""",
    )

    assert updated == [
        ("m_002", "用户使用 FastAPI、异步 Redis 和 ChromaDB"),
        ("m_003", "用户希望回复中展示上下文窗口占用比例"),
    ]
    assert manager._apply_user_memory_operations(updated, "Nothing to save") == updated


def test_user_memory_file_round_trip_and_limit_validation(tmp_path):
    manager = MemoryManager.__new__(MemoryManager)
    manager._user_memory_dir = tmp_path
    entries = [("m_001", "用户偏好中文技术说明，并保留关键英文术语")]

    manager._write_user_memory_entries("user-1", entries)

    assert manager._read_user_memory_entries("user-1") == entries
    assert manager._entries_fit_user_memory_limits(
        [("m_001", "x" * (manager.USER_MEMORY_MAX_ENTRY_CHARS + 1))]
    ) is False


def test_completed_turn_requests_a_review_every_five_turns():
    class FakeRedis:
        def __init__(self):
            self.counts = {}
            self.expirations = []

        async def incr(self, key):
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

        async def expire(self, key, seconds):
            self.expirations.append((key, seconds))

    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = FakeRedis()

    due = [
        asyncio.run(manager.record_completed_turn("u1", "c1"))
        for _ in range(10)
    ]

    assert due == [False, False, False, False, True, False, False, False, False, True]
    assert len(manager._redis.expirations) == 10
