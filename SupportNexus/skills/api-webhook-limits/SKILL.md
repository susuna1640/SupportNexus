---
id: api-webhook-limits
name: Webhook 与调用限流支持
description: 处理 Webhook 签名/回调问题、429 限流、配额与并发策略。
when_to_use: 用户提到 Webhook、回调、签名、429、限流、QPS、并发、调用额度或重试时。
agents: api
enabled: true
---

## Objective

提供安全、可复现的 Webhook 或限流排查路径，不把推断当成服务端事实。

## Gather

- 回调地址、环境、失败时间、状态码、响应摘要和 `request_id`。
- 签名算法、时间戳、是否使用原始请求体；不得收集完整 secret。
- 当前 QPS、并发、重试频率、调用量和套餐/额度信息。

## Workflow

1. Webhook 签名失败时，检查原始请求体、时间戳、环境和回调地址。
2. 429 时，先限制并发并使用指数退避，避免无间隔重试。
3. 区分短时突发、持续超额和套餐权限不足。
4. 给出验证方式和需要由订阅或企业支持核验的额度事项。

## Escalate

回调持续失败影响生产、需提升限额或需核验企业套餐时，附带时间窗口和 `request_id` 转对应支持。

## Boundaries

不得建议关闭签名校验、绕过鉴权或无限重试；不得要求用户发送完整 webhook secret。
