# SupportNexus 服务端

> 面向 B2B SaaS 客服与运营支持的、可观测的多 Agent 编排运行时。

SupportNexus 不把客服问题直接交给单一聊天模型，而是将上下文、请求分析、领域路由、知识检索、业务规则和运行反馈组织成一条可解释的服务链路。它适合处理 API 接入、产品上手、企业账户、技术故障和订阅账单等复杂支持请求。

本目录包含 Python 服务端及其运行依赖。项目总览和 Vue 工作台入口见 [仓库根 README](../README.md)；前端有独立的 [README](../SupportNexusFrontend/README.md)。

## 能力一览

- **三路请求分析**：融合 LLM、Embedding 与规则匹配，输出细粒度意图、领域分数、实体和置信度来源。
- **阈值式主/辅 Agent 编排**：领域最高分选择主 Agent；其他超过阈值的领域并行调用辅助 Agent 并合并结果。
- **按需 RAG**：以 ChromaDB 稠密检索、BM25 与 RRF 融合为基础，可选接入 Qwen 重排。
- **分层记忆**：文件全量加载长期用户事实；Redis 保存当前活跃阶段的摘要和最近消息；空闲 24 小时的阶段会结算至 ChromaDB，并按语义跨阶段召回。
- **动态 Skills**：从 `skills/` 加载业务处理规范，可通过 API 热重载。
- **工具治理**：知识库检索具备缓存、超时、熔断和降级；领域工具采用显式白名单。
- **质量闭环**：提供 Prometheus 指标、工具调用追踪、可选 Langfuse 观测，以及意图和端到端回复质量评测。

## 请求如何流转

```text
POST /chat
  │
  ├─ 检查 Redis 会话阶段空闲时间；超 24h 时先归档摘要至 ChromaDB 并清理 Redis
  ├─ 读取 Redis 当前阶段记忆、文件式长期记忆和 ChromaDB 语义历史
  ├─ 识别 intent / intent_group / domain_scores / entities
  ├─ 按领域分数和阈值生成可解释的主/辅 Agent 路由决策
  ├─ 向已路由 Agent 展示专属 Skill 目录；Agent 按需读取场景 SOP、调用 RAG 与领域工具
  ├─ 生成单 Agent 回复，或并行执行后合并结果
  └─ 写回记忆，并记录工具追踪、监控指标和可选链路观测
```

## Agent 角色

| Agent | 主要处理范围 |
| --- | --- |
| `general` | 通用咨询、需求澄清、基础客服分流 |
| `api` | API Key、鉴权、请求结构、错误码与接入排障 |
| `onboarding` | Workspace 建立、首次接入、功能上手与配置清单 |
| `enterprise` | 企业账户、角色权限、SSO、SCIM、合同类核验 |
| `technical` | 登录、故障、限流、环境与诊断路径 |
| `billing` | 订阅、用量、账单、退款与金额核验 |

每类 Agent 都有独立的角色契约、输入/输出边界和工具白名单。没有领域分数达到主路由阈值时，系统交给 `general` 处理通用咨询或澄清需求；系统不会声称已创建工单、转接人工或执行未经授权的业务操作。

默认主路由阈值为 `0.50`，辅助路由阈值为 `0.45`，每次最多并行两个辅助 Agent。路由器不再重复解析关键词或实体，只消费请求分析层输出的 `domain_scores`。

## 项目结构

```text
SupportNexus/
├── api/                 FastAPI 入口与 HTTP 接口
├── agents/              Agent 角色、路由、响应合并和领域工具
├── core/                意图识别、Skills 加载、LLM 与可观测性封装
├── mcp/                 项目内工具管理、知识库与重排实现
├── memory/              Redis / ChromaDB / 文件式记忆管理
├── monitor/             在线指标、告警与运行摘要
├── evaluation/          意图评测与端到端 LLM-as-Judge 评测
├── skills/              可热加载的业务规则
├── data/                演示文档、评测集和运行期持久化目录
├── tests/               单元与集成边界测试
└── config/              Prometheus 配置
```

## 快速开始

### 前置条件

- Python 3.12。
- 可用的 Anthropic API Key，或一个兼容 Anthropic Messages API 的模型服务。
- 可选：DashScope 账号（远程 Embedding 或 Qwen 重排）和 Langfuse（链路观测）。

### 1. 创建本地环境变量

`.env` 含密钥且不会随仓库分发；请在当前目录手动创建。最小配置如下：

```env
# 必填
ANTHROPIC_API_KEY=your_api_key

# 可选：官方 Anthropic 服务通常无需设置 BASE_URL。
# 使用兼容服务时，按服务商文档填写模型和地址。
# ANTHROPIC_MODEL=your_model
# ANTHROPIC_BASE_URL=https://your-compatible-provider.example

# Redis 密码；请使用强密码。
REDIS_PASSWORD=change_me
```

不要把 `.env`、模型密钥或真实用户数据提交到版本库。

### 2. Docker 构建状态

本目录已包含后端镜像定义 `Dockerfile` 与构建上下文规则 `.dockerignore`。仓库根目录的 `compose.yaml` 当前编排前端 Nginx、后端、Redis 和 ChromaDB。

### 3. 验证并发起第一轮对话

```bash
curl http://localhost:8000/health
```

首次使用一个从未被手动清空的空知识库时，服务会自动导入内置 B2B SaaS 示例文档。可检查当前知识库状态：

```bash
curl http://localhost:8000/knowledge/stats
```

如果曾显式清空知识库且希望恢复示例文档，可在知识库仍为空时调用 `POST /knowledge/seed-defaults`。该接口在知识库已有数据时会返回冲突，避免意外重复导入。

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "调用 /v1/chat 返回 401，怎样排查？",
    "user_id": "demo-user"
  }'
```

服务启动后可访问：

| 服务 | 地址 | 说明 |
| --- | --- | --- |
| SupportNexus API | `http://localhost:8000` | 主 API |
| Swagger UI | `http://localhost:8000/docs` | 交互式接口文档 |
| ChromaDB | `http://localhost:8001` | 向量数据库 |
| Redis | `localhost:6379` | 会话工作记忆 |
| Prometheus | `http://localhost:9092` | 指标查询 |

## 核心 API

Swagger 是完整且实时的接口契约。以下为常用入口：

| 类别 | 接口 | 用途 |
| --- | --- | --- |
| 运行状态 | `GET /health`、`GET /monitor`、`GET /metrics` | 健康检查、运行摘要和 Prometheus 指标 |
| 对话 | `POST /chat` | 触发记忆、意图识别、路由、Agent 执行和记忆回写 |
| Skills | `GET /skills`、`POST /skills/reload` | 查看或热重载业务规则 |
| 工具追踪 | `GET /trace/tool/{request_id}`、`GET /trace/tools` | 查看某次或近期请求的工具调用 |
| RAG 检索 | `POST /search`、`POST /search/baseline` | 对比优化链路与单路向量基线 |
| 知识库 | `POST /knowledge/add`、`POST /knowledge/upload`、`GET /knowledge/stats` | 添加、上传与查看知识库 |
| 知识库维护 | `GET /knowledge/chunks`、`DELETE /knowledge/chunks?title=...` | 查看或按标题删除文本块 |
| 初始化 | `POST /knowledge/seed-defaults` | 向空知识库导入内置示例 |
| 质量评测 | `POST /eval/run` | 运行意图与端到端质量评测 |

`POST /knowledge/upload` 支持 `.txt`、`.md` 与 JSON 文档数组，单文件上限为 10 MB。`DELETE /knowledge/chunks/all?confirm=true` 会清空全部知识库文本块，应仅在确认数据可丢弃时调用。

## 配置与降级行为

| 目的 | 变量 | 说明 |
| --- | --- | --- |
| 模型调用 | `ANTHROPIC_API_KEY` | 必填；缺失时服务无法完成初始化 |
| 兼容模型服务 | `ANTHROPIC_MODEL`、`ANTHROPIC_BASE_URL` | 选择模型或指向 Anthropic 兼容端点 |
| Redis | `REDIS_PASSWORD`、`REDIS_URL` | 配置 Redis 连接地址与认证信息 |
| Skills | `SUPPORT_NEXUS_SKILLS_DIR`、`SUPPORT_NEXUS_SKILLS_MAX_PROMPT_CHARS` | 覆盖规则目录及注入上限 |
| 长期记忆 | `SUPPORT_NEXUS_USER_MEMORY_DIR` | 覆盖文件式长期记忆目录 |
| 远程 Embedding | `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 或 `DASHSCOPE_EMBEDDING_ENDPOINT` | 可选接入 DashScope 兼容 Embedding 服务 |
| RAG 重排 | `DASHSCOPE_RERANK_ENDPOINT`、`SUPPORT_NEXUS_RAG_RERANK_MODEL` | 可选指定 Qwen 重排端点与模型 |
| Langfuse | `LANGFUSE_BASE_URL`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_TRACING_ENABLED` | 配齐密钥后启用链路观测；可设为 `false` 关闭 |
| 内容采集 | `LANGFUSE_CAPTURE_CONTENT=false` | 只保留结构化观测，不上传消息、Prompt、工具结果和回答正文 |

未配置 DashScope，或远程 Embedding 调用失败时，意图识别会退化为本地 256 维字符 n-gram 哈希向量；可使用 `SUPPORT_NEXUS_INTENT_EMBEDDING_STRICT=true` 改为严格报错。未配置重排服务时，RAG 仍会返回混合召回结果，并保留 RRF 排序。

## 数据、运行与安全

| 数据 | 默认用途 | 持久化位置 |
| --- | --- | --- |
| Redis | 当前活跃阶段的工作记忆（逻辑空闲阈值 24h，结算前保留宽限） | 由 Redis 服务配置 |
| ChromaDB | 知识库与已结算会话阶段的情景摘要 | 由 ChromaDB 服务配置 |
| `data/user_memories/` | 长期用户事实 | 本地目录 |
| `data/eval/` | 评测基线与数据集 | 本地目录 |

- API 默认允许跨域请求。若部署到公网，请在反向代理或应用层收紧来源、认证、限流和 TLS 配置。
- `/eval/run` 会执行模型调用，可能产生费用和一定耗时；建议在隔离的评测数据与预算下运行。
- 工具不会伪造支付、订单、合同或权限系统的读写结果。需要真实业务动作时，应在经过认证、授权和审计的集成层实现。

## 本地开发与测试

Docker 学习完成前，可在宿主机自行运行 Redis 和 ChromaDB，再安装依赖并配置对应连接地址（把 `change_me` 换成 `.env` 中的实际 Redis 密码）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
REDIS_URL=redis://:change_me@localhost:6379/0 \
CHROMA_HOST=localhost \
CHROMA_PORT=8001 \
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

测试工具不属于运行时依赖；需要执行测试时另行安装：

```bash
pip install pytest
python -m pytest tests
```

## 延伸阅读

- [Skills 说明](skills/README.md)
- [HTTP API 实现](api/main.py)
- [Agent 编排实现](agents/agent_orchestrator.py)
- [记忆管理实现](memory/conversation_memory.py)
- [知识库与混合检索实现](mcp/knowledge_base.py)
