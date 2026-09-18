# SupportNexus：面向 B2B SaaS 客服与运营支持的多 Agent 编排平台

这是 **SupportNexus：面向 B2B SaaS 客服与运营支持的多 Agent 编排平台** 的 Vue 前端工作台。它用于发起和调试客户问题，并将后端实际返回的意图、领域分数、Agent 路由、知识库命中与运行信息集中展示。

## 适用场景

可用以下 B2B SaaS 客服/运营问题进行联调或演示：

- API Key、鉴权失败、错误码和请求 ID 排查；
- workspace 创建、成员邀请、角色与权限配置；
- 企业账户、SSO、SCIM 等接入咨询；
- Webhook 签名校验、回调失败等集成故障；
- 订阅、账单、重复扣款，以及需要多个专业 Agent 协同的复合问题。

## 页面能力

- **对话工作台**：发送客户问题，并保存用户 ID 与会话 ID；后端返回对应字段时，展示业务意图、意图组、领域分数、实体、主/辅 Agent、路由原因与置信度、知识库使用情况和响应耗时。
- **工具调用追踪**：响应包含 `request_id` 且后端提供追踪数据时，展示该请求的工具调用记录。
- **知识库运营**：查看知识块统计、检索知识、添加文本知识，以及上传知识库文件。
- **运行与质量检查**：查看运行监控摘要、已加载的 Skills、重新加载 Skills，并触发后端内置评测。
- **连接配置**：连接 Python 后端，并检查当前连接的健康状态。

## 后端与接口

### 默认后端地址

Python 后端默认地址为 `http://localhost:8000`。

开发模式下，Vite 会代理：

| 前端路径 | 代理到 |
| --- | --- |
| `/api/python` | `http://localhost:8000` |

前端会向 Python 后端请求以下接口：

| 方法 | 接口 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/chat` | 发送客服问题；请求体包含 `message`、`user_id`、可选的 `conv_id` |
| `GET` | `/monitor` | 读取监控摘要 |
| `GET` / `POST` | `/skills`、`/skills/reload` | 查看或重新加载 Skills |
| `GET` | `/knowledge/stats` | 查询知识库统计 |
| `POST` | `/search?query=...&top_k=...` | 检索知识库 |
| `POST` | `/knowledge/add`、`/knowledge/upload` | 添加文本知识或上传文件 |
| `POST` | `/eval/run` | 运行评测 |
| `GET` | `/trace/tool/{request_id}` | 查询工具调用追踪 |

聊天响应会兼容 `conv_id` / `conversation_id`、`agent_type`、`latency_ms` 等字段，也会在后端提供时读取 `intent_group`、`domain_scores`、`primary_agent`、`supporting_agents`、`entities` 与 `routing_reason` 等编排信息；缺失字段不会由前端伪造。

页面右上角的“API 文档”会打开 Python 后端的 `/docs`。

## 本地运行

在 `SupportNexusFrontend` 目录中安装依赖：

```bash
npm install
```

启动开发服务器：

```bash
npm run dev
```

访问：

```text
http://localhost:5173
```

如果后端端口不是默认值，可以在启动时覆盖：

```bash
VITE_PYTHON_API_URL=http://localhost:8000 \
npm run dev
```

## 后端启动参考

Python 后端默认地址为 `http://localhost:8000`，启动后即可连接前端工作台。
