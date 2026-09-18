# SupportNexus

面向 B2B SaaS 客服与运营支持的多 Agent 编排项目。仓库由 Python 服务端和 Vue 前端工作台组成：前端向服务端发送客户问题、管理知识库并展示运行结果；服务端负责意图识别、Agent 路由、知识检索、记忆与评测。

## 当前包含的内容

### 服务端：`SupportNexus/`

- FastAPI HTTP API，提供健康检查、对话、知识库、Skills、工具追踪、监控和评测接口；完整接口可在服务启动后访问 `/docs`。
- 6 类支持 Agent：通用咨询、API 接入、上手引导、企业账户、技术支持和账单订阅。
- 请求分析与路由：结合 LLM、Embedding 和规则匹配产生意图、领域分数与实体；按分数选择主 Agent，并可附加辅助 Agent。
- 知识库检索：使用 ChromaDB 向量检索、BM25 与 RRF 融合；可选接入重排服务。
- 会话与用户记忆：Redis 保存活跃会话，ChromaDB 保存已结算会话摘要，本地文件保存长期用户事实。
- 可热重载的业务规则：`SupportNexus/skills/` 中包含 API、账单、企业账户、上手与技术诊断等场景的规则文档。
- 运行观测与评测：提供 Prometheus 指标、工具调用追踪、可选 Langfuse 链路观测，以及意图和端到端评测代码。

### 前端：`SupportNexusFrontend/`

- Vue + Vite 工作台；开发模式默认连接 `http://localhost:8000`，Docker 模式由 Nginx 转发至后端容器。
- 对话页面会展示服务端返回的意图、领域分数、主/辅 Agent、路由原因、知识库使用情况、耗时与工具调用追踪。
- 支持 Markdown 回复渲染。
- 包含知识库统计、检索、文本添加、文件上传、Skills 重载、运行监控与评测触发页面。

## 目录结构

```text
.
├── SupportNexus/             Python 服务端
│   ├── api/                  FastAPI 入口和接口
│   ├── agents/               Agent 编排与工具定义
│   ├── core/                 意图识别、LLM、Skills 与观测封装
│   ├── mcp/                  知识库、重排和工具管理
│   ├── memory/               会话与用户记忆
│   ├── evaluation/           质量评测
│   ├── monitor/              运行指标与告警
│   ├── skills/               业务场景规则
│   └── tests/                测试
├── SupportNexusFrontend/     Vue 前端工作台
├── compose.yaml              四服务的 Docker 编排清单
├── SupportNexus/Dockerfile   后端镜像构建配方
├── SupportNexusFrontend/Dockerfile  前端构建与 Nginx 运行镜像配方
├── .githooks/                Git 提交前敏感信息检查
└── .gitignore                本地密钥、运行数据与构建产物的忽略规则
```

## 使用 Docker 运行整套项目

Docker Compose 会按根目录的 `compose.yaml` 一次启动四个服务：

```text
浏览器
  │ http://127.0.0.1:8080
  ▼
frontend（Nginx，提供 Vue 构建产物）
  │ /api/python/*
  ▼
backend（FastAPI）
  ├── redis（活跃会话记忆）
  └── chromadb（知识库与会话摘要）
```

Redis 和 ChromaDB 仅在 Docker 内部网络中可访问；浏览器只能访问前端 `8080` 端口，后端 `8000` 端口也只绑定到本机回环地址。

### 1. 准备本机配置

Docker Desktop 需要处于运行状态。首次启动前，基于模板创建服务端配置，并填写所需的模型服务密钥或兼容服务地址：

```bash
cp SupportNexus/.env.example SupportNexus/.env
```

`.env` 只保留在本机，既不会被 Git 提交，也不会被复制进镜像。

### 2. 构建并启动

在仓库根目录执行：

```bash
docker compose up --build -d
```

这条命令会先构建 Python 后端镜像和 Vue 前端镜像，再按健康检查顺序启动 Redis、ChromaDB、后端和前端。首次构建需要下载基础镜像与依赖，后续没有变更的层会复用 Docker 缓存。

若网络需要代理，Docker Desktop 会自动读取 macOS 当前的系统代理；可先验证 Docker 能否拉取前端需要的基础镜像：

```bash
docker pull node:22-alpine
```

### 3. 访问与验证

| 入口 | 地址 |
| --- | --- |
| 前端工作台 | `http://127.0.0.1:8080` |
| 后端 API | `http://127.0.0.1:8000` |
| Swagger API 文档 | `http://127.0.0.1:8000/docs` |

查看所有容器是否健康：

```bash
docker compose ps
```

验证浏览器入口、Nginx 反向代理和后端 API 的完整链路：

```bash
curl -fsS http://127.0.0.1:8080/api/python/health \
  | jq '{status, agent_count: (.agents | length)}'
```

预期返回 `status: "ok"`，并显示当前已加载的 Agent 数量。

### 4. 日常管理

```bash
# 查看前端与后端的实时日志
docker compose logs -f frontend backend

# 停止并移除容器、网络；Redis 与 ChromaDB 数据卷会保留
docker compose down

# 同时删除容器数据卷（会不可恢复地清空 Redis、ChromaDB 与本地持久化数据）
docker compose down -v
```

Docker 配置文件的职责分别是：`Dockerfile` 描述单个服务镜像如何构建；`.dockerignore` 防止密钥和本地运行数据进入构建上下文；`compose.yaml` 描述多服务之间的网络、端口、数据卷、依赖关系和健康检查。

## 本地运行

### 1. 配置服务端

服务端需要可用的模型服务；Redis 和 ChromaDB 用于运行期记忆与知识库。创建本机配置文件时，使用模板而不是提交真实密钥：

```bash
cd SupportNexus
cp .env.example .env
```

填写 `.env` 后，在已准备好 Redis 和 ChromaDB 的本机环境中启动 API：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

服务端默认地址为 `http://localhost:8000`，接口文档为 `http://localhost:8000/docs`。

### 2. 启动前端

在另一个终端运行：

```bash
cd SupportNexusFrontend
npm install
npm run dev
```

访问 `http://localhost:5173`。开发服务器会将 `/api/python` 代理到服务端；也可以通过 `VITE_PYTHON_API_URL` 覆盖后端地址。

## 当前运行边界

- Docker 运行所需的前后端 `Dockerfile`、各自的 `.dockerignore` 与根目录 `compose.yaml` 已包含在仓库中。
- `.env`、虚拟环境、运行数据、评测输出和私钥文件均被 Git 忽略；请始终从 `SupportNexus/.env.example` 创建本机 `.env`，不要提交真实密钥。
- 本项目内的知识库、记忆和评测数据使用本地或已配置的外部服务；请勿向其中写入不应被用于调试或评测的真实用户数据。

## 详细说明

- [服务端 README](SupportNexus/README.md)
- [前端 README](SupportNexusFrontend/README.md)
- [业务规则（Skills）说明](SupportNexus/skills/README.md)
