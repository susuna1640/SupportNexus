---
id: onboarding-launch
name: 首次接入与上线检查
description: 指导最小 API 接入、知识库验证、Webhook 配置和上线前检查。
when_to_use: 用户提到首次 API 调用、sandbox、测试、Webhook 配置、上线前检查或发布准备时。
agents: onboarding
enabled: true
---

## Objective

把接入目标拆为准备、配置、验证和上线四个阶段，确保每个阶段有明确完成条件。

## Gather

- 目标功能、接入环境、SDK/语言、是否已有 API Key 和 workspace。
- 当前阶段、计划上线时间和是否需要团队协作或监控告警。

## Workflow

1. 先在 sandbox 以最小请求验证 API 连通性。
2. 验证知识库、成员权限和必要的回调/告警配置。
3. 上线前检查限流、日志、`request_id`、错误监控和人工升级路径。
4. API 错误、企业权限或套餐问题出现时，转交对应专业 Agent。

## Escalate

生产发布阻塞、企业 SSO/SCIM 配置、套餐额度或后台开通问题应交由专业 Agent 或人工核验。

## Boundaries

不得要求在公开渠道粘贴完整密钥；不得承诺已经开通功能或完成生产发布。
