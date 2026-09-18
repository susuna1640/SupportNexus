# SupportNexus Skills 文档

SupportNexus 启动时会从 `SUPPORT_NEXUS_SKILLS_DIR` 读取 Skills。路由选定 Agent 后，系统只把该 Agent 可用的 Skill **名称、说明与适用场景**放入其 system prompt；Agent 需要细分 SOP 时，再调用受权限约束的 `load_agent_skill` 读取正文。Skills 适合维护业务处理规范、客服话术、技术排障 SOP、账单审核边界、升级规则和禁止事项。

当前内置 11 份场景化 Skills；每个专业 Agent 最多两份，GeneralAgent 一份：

```text
skills/general-triage/SKILL.md                 # general：澄清、分流、投诉与跨域拆分
skills/api-access-auth/SKILL.md                # api：接入、鉴权、权限与请求结构
skills/api-webhook-limits/SKILL.md             # api：Webhook、签名、限流与配额
skills/onboarding-workspace/SKILL.md           # onboarding：workspace 与团队初始化
skills/onboarding-launch/SKILL.md              # onboarding：首次接入与上线检查
skills/enterprise-access/SKILL.md              # enterprise：角色、SSO/SCIM 与审计治理
skills/enterprise-commercial/SKILL.md          # enterprise：合同、采购与续费流程
skills/technical-diagnosis/SKILL.md            # technical：登录、错误码与平台故障
skills/technical-deployment/SKILL.md           # technical：部署、配置与性能
skills/billing-transactions/SKILL.md           # billing：退款、扣款与支付核验
skills/billing-subscription-invoice/SKILL.md   # billing：订阅、用量与发票
```

## Skill 文件格式

每个 Skill 使用独立目录和 `SKILL.md`：

```text
skills/<skill-id>/SKILL.md
```

文件顶部使用标准 front matter：

```markdown
---
id: api-access-auth
name: API 接入与鉴权排障
description: 处理接口接入、请求结构、鉴权、权限和常见状态码问题。
when_to_use: 用户提到 endpoint、SDK、请求失败、401、403、404、请求体或 request_id 时。
agents: api
enabled: true
---
```

字段说明：

- `id`：稳定且全局唯一的 Skill 标识；Agent 通过它调用 `load_agent_skill`。
- `name`：Skill 展示名称。
- `description`：简短说明，帮助 Agent 从目录中选择，也便于 `/skills` 排查。
- `when_to_use`：一句话描述适用场景，供 Agent 按需选择。
- `agents`：适用 Agent。服务端会据此阻止跨 Agent 加载。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 一份 Skill 只描述一个可独立触发的场景，不混入其他领域的规则。
- 正文统一使用 `Objective`、`Gather`、`Workflow`、`Escalate`、`Boundaries` 五个章节。
- Agent 角色职责、工具范围和通用安全边界保留在 system prompt，不放进每份 Skill 重复维护。
- 描述必须能让对应 Agent 判断何时加载，不依赖全局关键词自动注入。
- 涉及支付、隐私、密码、验证码、API Key、Token 等敏感信息时，明确禁止收集或公开。
- 无法保证的事项使用“通常”“预计”“需要核验后确认”等保守表述，并写清人工或二线升级条件。

## 热加载与按需读取

修改 Skill 文件后无需重启服务：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看已加载的 Skill 目录和解析错误：

```bash
curl http://localhost:8000/skills
```

`/skills` 只返回目录摘要，不默认暴露 Skill 正文。单个 Agent 在一次请求中最多加载两份已授权 Skill，且每份正文最多占总 Skill 预算的一半。
