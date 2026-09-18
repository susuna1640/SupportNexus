"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储情景记忆；长期用户记忆由文件保存
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"
    RRF_K = 60
    USER_CLEARED_KEY = "support-nexus_user_cleared"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        if chromadb is None:
            raise RuntimeError("未安装 chromadb；请先安装 requirements.txt 中的依赖")
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "SupportNexus RAG 知识库"},
        )
        self._bm25_doc_count = -1
        self._bm25_documents: List[Dict[str, Any]] = []
        self._bm25_term_frequencies: List[Counter] = []
        self._bm25_document_frequencies: Counter = Counter()
        self._bm25_avg_doc_len = 0.0

        # 首次启动可导入默认文档；通过管理接口清空后保持为空，等待用户主动导入。
        collection_metadata = self._collection.metadata or {}
        if self._collection.count() == 0 and not collection_metadata.get(self.USER_CLEARED_KEY):
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            self._invalidate_bm25_index()
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items: List[Dict[str, Any]] = []
        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        ids = (results.get("ids") or [[]])[0] or []
        if documents:
            for index, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                items.append({
                    "id":       ids[index] if index < len(ids) else self._result_id(doc, meta),
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                    "retrieval": "dense",
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    def search_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """BM25 词法召回，适合精确术语、错误码和套餐名等关键词。"""
        if top_k <= 0:
            return []
        self._ensure_bm25_index()
        if not self._bm25_documents:
            return []

        query_terms = Counter(self._tokenize(query))
        if not query_terms:
            return []

        k1, b = 1.5, 0.75
        total_docs = len(self._bm25_documents)
        scores = [0.0] * total_docs
        for term, query_tf in query_terms.items():
            document_frequency = self._bm25_document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue
            idf = math.log(1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            for index, term_frequencies in enumerate(self._bm25_term_frequencies):
                term_frequency = term_frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                doc_len = sum(term_frequencies.values())
                denominator = term_frequency + k1 * (
                    1.0 - b + b * doc_len / max(self._bm25_avg_doc_len, 1.0)
                )
                scores[index] += query_tf * idf * term_frequency * (k1 + 1.0) / denominator

        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0.0),
            key=lambda index: (-scores[index], self._bm25_documents[index]["id"]),
        )[:top_k]
        return [
            {
                **self._bm25_documents[index],
                "score": round(scores[index], 6),
                "retrieval": "bm25",
            }
            for index in ranked_indices
        ]

    async def search_bm25_async(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search_bm25, query, top_k)

    def search_hybrid(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """同步混合召回：Chroma 稠密检索 + BM25，再以 RRF 融合。"""
        return self._fuse_hybrid_results(
            self.search(query, top_k),
            self.search_bm25(query, top_k),
            top_k,
        )

    async def search_hybrid_async(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """并行执行稠密、稀疏召回，再以 Reciprocal Rank Fusion 融合。"""
        dense, sparse = await asyncio.gather(
            self.search_async(query, top_k),
            self.search_bm25_async(query, top_k),
        )
        return self._fuse_hybrid_results(dense, sparse, top_k)

    def list_chunks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出入库的文本块，供知识库管理和导入核验使用。"""
        raw = self._collection.get(
            include=["documents", "metadatas"],
            limit=max(1, limit),
            offset=max(0, offset),
        )
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        return [
            {
                "id": str(chunk_id),
                "title": (metadata or {}).get("title", ""),
                "content": str(content or ""),
                "chunk": (metadata or {}).get("chunk_index", 0),
                "total_chunks": (metadata or {}).get("total_chunks", 1),
            }
            for chunk_id, content, metadata in zip(ids, documents, metadatas)
        ]

    async def list_chunks_async(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.list_chunks, limit, offset)

    def delete_chunks_by_title(self, title: str) -> int:
        """删除某个原文标题下的全部 chunk。"""
        raw = self._collection.get(where={"title": title}, include=["metadatas"])
        ids = raw.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
            self._invalidate_bm25_index()
            if self._collection.count() == 0:
                self._mark_user_cleared()
        return len(ids)

    async def delete_chunks_by_title_async(self, title: str) -> int:
        return await asyncio.to_thread(self.delete_chunks_by_title, title)

    def clear_chunks(self) -> int:
        """只清空 knowledge_base collection，不影响对话记忆 collection。"""
        raw = self._collection.get(include=["metadatas"])
        ids = raw.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
            self._invalidate_bm25_index()
        self._mark_user_cleared()
        return len(ids)

    async def clear_chunks_async(self) -> int:
        return await asyncio.to_thread(self.clear_chunks)

    def seed_default_documents(self) -> int:
        """向空知识库显式导入项目内置文档，避免依赖服务启动时机。"""
        if self._collection.count() > 0:
            raise ValueError("知识库非空；请先清空或使用上传接口导入新文档")
        self._load_default_docs()
        return self._collection.count()

    async def seed_default_documents_async(self) -> int:
        return await asyncio.to_thread(self.seed_default_documents)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    async def hybrid_search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """作为 MCP 工具 handler 注册的混合检索入口。"""
        query = params.get("query", "")
        top_k = params.get("top_k", 20)
        return await self.search_hybrid_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _invalidate_bm25_index(self) -> None:
        self._bm25_doc_count = -1

    def _mark_user_cleared(self) -> None:
        """让显式清空后的 collection 在下次启动时不再自动回填默认样例。"""
        try:
            metadata = dict(self._collection.metadata or {})
            metadata[self.USER_CLEARED_KEY] = True
            self._collection.modify(metadata=metadata)
        except Exception as ex:
            logger.warning("无法记录知识库已清空状态: %s", ex)

    def _ensure_bm25_index(self) -> None:
        current_count = self._collection.count()
        if self._bm25_doc_count == current_count:
            return

        raw = self._collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        documents_for_index: List[Dict[str, Any]] = []
        term_frequencies: List[Counter] = []
        document_frequencies: Counter = Counter()

        for doc_id, content, metadata in zip(ids, documents, metadatas):
            content = str(content or "")
            metadata = metadata or {}
            tokens = self._tokenize(content)
            frequencies = Counter(tokens)
            documents_for_index.append({
                "id": str(doc_id),
                "title": metadata.get("title", ""),
                "content": content,
                "chunk": metadata.get("chunk_index", 0),
            })
            term_frequencies.append(frequencies)
            document_frequencies.update(frequencies.keys())

        self._bm25_doc_count = current_count
        self._bm25_documents = documents_for_index
        self._bm25_term_frequencies = term_frequencies
        self._bm25_document_frequencies = document_frequencies
        total_length = sum(sum(frequencies.values()) for frequencies in term_frequencies)
        self._bm25_avg_doc_len = total_length / len(term_frequencies) if term_frequencies else 0.0
        logger.info("BM25 索引已刷新: %d 个文档片段", len(documents_for_index))

    def _fuse_hybrid_results(
        self,
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        for source, results in (("dense", dense), ("bm25", sparse)):
            for rank, item in enumerate(results, start=1):
                key = str(item.get("id") or self._result_id(item.get("content", ""), item))
                combined = fused.setdefault(key, dict(item))
                combined["rrf_score"] = combined.get("rrf_score", 0.0) + 1.0 / (self.RRF_K + rank)
                combined[f"{source}_rank"] = rank
                combined[f"{source}_score"] = item.get("score", 0.0)

        ranked = sorted(
            fused.values(),
            key=lambda item: (-item["rrf_score"], str(item.get("id", ""))),
        )[:max(top_k, 0)]
        for item in ranked:
            item["score"] = round(item["rrf_score"], 6)
            item["retrieval"] = "hybrid_rrf"
        return ranked

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = text.lower()
        tokens = re.findall(r"[a-z0-9_./-]+", normalized)
        for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
            tokens.extend(sequence)
            tokens.extend(sequence[index:index + 2] for index in range(len(sequence) - 1))
        return tokens

    @staticmethod
    def _result_id(content: Any, metadata: Any) -> str:
        return hashlib.md5(f"{content}|{metadata}".encode("utf-8")).hexdigest()

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（B2B SaaS Developer Support 场景）。"""
        default_docs = [
            {
                "title": "API 接入指南",
                "content": (
                    "SupportNexus API 接入指南。"
                    "开发者需要在控制台 Settings - API Keys 生成 API Key，并只保存在服务端环境。"
                    "对话接口为 POST /v1/chat，请求体包含 message、user_id、conv_id 和可选 metadata。"
                    "响应包含 response、intent、primary_agent、supporting_agents 和 tool_traces。"
                    "调试失败时应记录 endpoint、HTTP 方法、状态码、业务错误码、request_id、发生时间和调用环境。"
                    "不要在公开渠道提供完整 API Key、Token 或 webhook secret。"
                ),
            },
            {
                "title": "API 错误码说明",
                "content": (
                    "API 错误码说明。"
                    "401 或 AUTH_001 表示 API Key 无效、缺失或已过期，需要检查 Authorization Bearer 格式和 key 所属 workspace。"
                    "403 或 PLAN_LIMIT_EXCEEDED 表示权限、套餐、IP 白名单或企业功能未开通，需要核验 workspace 权限和套餐范围。"
                    "429 或 RATE_LIMIT_EXCEEDED 表示 QPS、并发或月度调用量超过限制，应使用指数退避并评估限额提升。"
                    "WEBHOOK_SIGNATURE_INVALID 表示 webhook 签名失败，应使用原始请求体、正确 secret 和有效 timestamp 计算签名。"
                    "INVALID_REQUEST_SCHEMA 表示请求体结构或字段类型不符合当前 API 版本。"
                ),
            },
            {
                "title": "订阅计划与账单",
                "content": (
                    "订阅计划与账单说明。"
                    "Starter 每月 99 元，支持 1000 次对话、1 个自定义 Agent 和标准 API 调用。"
                    "Pro 每月 499 元，支持 10000 次对话、3 个自定义 Agent、Webhook、记忆管理和评测报告。"
                    "Enterprise 为定制定价，支持无限对话、自定义 Agent 数量、SSO/SCIM、审计日志、专属技术支持和合同续费流程。"
                    "账单争议需要核验 workspace_id、套餐名称、账单周期、席位数量、用量范围、金额和支付渠道。"
                    "取消自动续费通常只影响下一周期，当前周期退款需要按订阅状态、实际用量和规则审核。"
                ),
            },
            {
                "title": "企业账户与权限",
                "content": (
                    "企业账户与权限说明。"
                    "企业 workspace 支持 owner、admin、member 和 billing admin 等角色。"
                    "成员邀请、角色调整、SSO/SCIM、审计日志、IP 白名单和管理员变更需要管理员权限。"
                    "SSO 配置需要企业套餐、IdP 元数据、回调地址、证书和 NameID 映射。"
                    "SCIM 同步需要确认 IdP、同步范围、用户属性映射和测试用户。"
                    "合同续费、报价、折扣、付款条款和安全审查需要客户成功或人工支持处理，Agent 不能直接承诺。"
                ),
            },
            {
                "title": "新用户接入流程",
                "content": (
                    "新用户接入流程。"
                    "第一步创建 workspace 并确认 owner/admin。"
                    "第二步在服务端生成 API Key，完成 sandbox 环境的最小 /v1/chat 请求。"
                    "第三步导入一份小型知识库并验证检索命中。"
                    "第四步邀请团队成员并配置角色权限。"
                    "第五步配置 webhook、错误监控、request_id 日志和人工升级路径。"
                    "上线前需要检查限流、重试、日志、安全边界和回滚方案。"
                ),
            },
            {
                "title": "平台技术排障",
                "content": (
                    "平台技术排障说明。"
                    "控制台 500 需要记录页面路径、浏览器版本、发生时间和 request_id。"
                    "控制台登录失败需要区分账号密码、验证码、SSO、账号状态和浏览器缓存。"
                    "导入知识库失败需要确认文件格式、大小、编码、切片结果和向量化任务状态。"
                    "生产环境持续失败、数据丢失、权限异常或多名成员受影响时，应升级到二线技术支持。"
                    "排障建议必须从低风险操作开始，不能要求用户删除数据、关闭安全校验或公开敏感密钥。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
