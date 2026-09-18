---
id: api-access-auth
name: API 接入与鉴权排障
description: 处理接口接入、请求结构、鉴权、权限和常见状态码问题。
when_to_use: 用户提到 endpoint、SDK、请求失败、401、403、404、请求体或 request_id 时。
agents: api
enabled: true
---

## Objective

基于可验证的请求信息，将问题定位到鉴权、权限、请求结构、环境或服务端异常的合理范围。

## Gather

- HTTP 方法、接口路径、环境和 SDK/语言版本。
- HTTP 状态码、业务错误码、响应摘要、发生时间和 `request_id`。
- 是否携带鉴权头；只接收脱敏后的 Key 或 Token 摘要。

## Workflow

1. 区分认证、授权、请求结构、资源不存在和服务端异常。
2. 优先解释错误码，并使用最小请求复现思路验证。
3. 从环境、凭据归属、签名/时间戳、权限和请求字段依次排查。
4. 给出 3–5 条低风险步骤与仍需补充的字段。

## Escalate

需要服务端日志、后台开通、套餐变更或确认平台故障时，提交 `request_id`、时间和影响范围给二线或对应业务支持。

## Boundaries

不得索要完整 API Key、Token、私钥或 secret；不得声称已查看后台日志、已开通权限或已修改配置。
