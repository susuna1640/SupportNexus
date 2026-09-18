from pathlib import Path

from agents.agent_orchestrator import APIAgent
from core.skill_loader import SkillManager


class FakeClient:
    pass


def _manager() -> SkillManager:
    root = Path(__file__).resolve().parents[1]
    manager = SkillManager(str(root / "skills"), max_prompt_chars=4_000)
    manager.load()
    return manager


def test_skill_catalog_is_scoped_to_the_selected_agent():
    manager = _manager()

    assert len(manager.skills) == 11
    assert [skill.id for skill in manager.catalog_for("api")] == [
        "api-access-auth",
        "api-webhook-limits",
    ]
    assert [skill.id for skill in manager.catalog_for("billing")] == [
        "billing-subscription-invoice",
        "billing-transactions",
    ]

    catalog = manager.catalog_prompt_for("api")
    assert "api-access-auth" in catalog
    assert "api-webhook-limits" in catalog
    assert "billing-transactions" not in catalog
    assert "## Workflow" not in catalog


def test_skill_body_can_only_be_loaded_by_its_owner_agent():
    manager = _manager()

    allowed = manager.load_for_agent("api-access-auth", "api")
    denied = manager.load_for_agent("api-access-auth", "billing")
    missing_agent = manager.load_for_agent("api-access-auth")

    assert allowed["success"] is True
    assert "## Workflow" in allowed["instructions"]
    assert denied["success"] is False
    assert "不属于 billing Agent" in denied["error"]
    assert missing_agent["success"] is False
    assert "必须提供当前 Agent 类型" in missing_agent["error"]


def test_skill_metadata_requires_a_unique_kebab_case_id(tmp_path):
    skill_dir = tmp_path / "invalid-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
id: Invalid_ID
name: 无效 Skill
description: 用于验证元数据校验。
when_to_use: 仅用于测试。
agents: api
enabled: true
---

## Objective
测试。
""",
        encoding="utf-8",
    )
    manager = SkillManager(str(tmp_path))

    manager.load()

    assert manager.skills == []
    assert len(manager.errors) == 1
    assert "kebab-case" in manager.errors[0]


def test_agent_exposes_the_scoped_skill_loader_only_at_execution_time():
    manager = _manager()
    agent = APIAgent(FakeClient(), "test-model", skill_manager=manager)

    assert "load_agent_skill" not in agent.get_tools()
    tools = agent._tools_for_request()
    result = tools["load_agent_skill"].handler(None, {"skill_id": "api-webhook-limits"})

    assert "load_agent_skill" in tools
    assert result["success"] is True
    assert result["skill_id"] == "api-webhook-limits"
