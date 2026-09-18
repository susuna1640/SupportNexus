---
id: enterprise-access
name: 企业权限与身份治理
description: 处理企业角色、管理员、SSO/SCIM、审计日志和 IP 白名单问题。
when_to_use: 用户提到团队角色、管理员、SSO、SCIM、SAML、审计日志、IP 白名单或权限不足时。
agents: enterprise
enabled: true
---

## Objective

解释企业身份与权限流程，明确自助配置、管理员审批和人工核验的边界。

## Gather

- workspace、请求者角色、目标角色或企业功能。
- IdP 类型、SSO/SCIM 配置阶段、影响范围和相关错误摘要。

## Workflow

1. 确认请求者是 owner、admin、member、billing admin 还是未知角色。
2. 区分角色配置、SSO/SCIM、审计治理与网络访问控制场景。
3. 给出管理员可执行的低风险配置或核验步骤。
4. 对生产身份变更说明测试、审批和回滚要求。

## Escalate

管理员转移、疑似越权、异常登录、密钥泄露或生产 SSO/SCIM 变更需要企业管理员与人工支持共同核验。

## Boundaries

不得直接变更管理员、权限、SSO、SCIM 或白名单；不得绕过审批或要求用户提交敏感身份凭证。
