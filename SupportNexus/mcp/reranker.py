"""DashScope Qwen3-Rerank client used by the optimized RAG pipeline."""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    items: List[Any]
    applied: bool
    model: str
    error: Optional[str] = None


class Qwen3Reranker:
    """Call DashScope's qwen3-rerank compatible API without a model-side dependency."""

    DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."

    def __init__(
        self,
        api_key: str,
        workspace_id: str = "",
        region: str = "cn-beijing",
        endpoint: str = "",
        model: str = "qwen3-rerank",
        timeout_s: float = 10.0,
        instruct: str = DEFAULT_INSTRUCT,
    ):
        self.api_key = api_key.strip()
        self.workspace_id = workspace_id.strip()
        self.region = region.strip() or "cn-beijing"
        self.endpoint = endpoint.strip() or self._workspace_endpoint()
        self.model = model.strip() or "qwen3-rerank"
        self.timeout_s = max(float(timeout_s), 0.1)
        self.instruct = instruct.strip()

    @classmethod
    def from_env(cls) -> "Qwen3Reranker":
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            workspace_id=os.getenv("DASHSCOPE_WORKSPACE_ID", ""),
            region=os.getenv("DASHSCOPE_REGION", "cn-beijing"),
            endpoint=os.getenv("DASHSCOPE_RERANK_ENDPOINT", ""),
            model=os.getenv("SUPPORT_NEXUS_RAG_RERANK_MODEL", "qwen3-rerank"),
            timeout_s=float(os.getenv("SUPPORT_NEXUS_RAG_RERANK_TIMEOUT_S", "10")),
            instruct=os.getenv("SUPPORT_NEXUS_RAG_RERANK_INSTRUCT", cls.DEFAULT_INSTRUCT),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.endpoint)

    def _workspace_endpoint(self) -> str:
        if not self.workspace_id:
            return ""
        return (
            f"https://{self.workspace_id}.{self.region}.maas.aliyuncs.com"
            "/compatible-api/v1/reranks"
        )

    async def rerank(self, query: str, items: List[Any], top_k: int) -> RerankResult:
        limit = max(1, top_k)
        if len(items) <= limit:
            return RerankResult(items=items[:limit], applied=False, model=self.model)
        if not self.configured:
            return RerankResult(
                items=items[:limit],
                applied=False,
                model=self.model,
                error="Qwen3-Rerank 未配置 DASHSCOPE_API_KEY 和 rerank endpoint/workspace",
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [self._document_text(item) for item in items],
            "top_n": limit,
        }
        if self.instruct:
            payload["instruct"] = self.instruct

        try:
            data = await self._post_json(payload)
            raw_results = data.get("results")
            if not isinstance(raw_results, list):
                message = str(data.get("message") or data.get("code") or "响应缺少 results")
                raise RuntimeError(message)

            ranked: List[Any] = []
            consumed = set()
            for raw in raw_results:
                index = raw.get("index") if isinstance(raw, dict) else None
                if not isinstance(index, int) or not 0 <= index < len(items) or index in consumed:
                    continue
                consumed.add(index)
                ranked.append(self._attach_score(items[index], raw.get("relevance_score"), index))

            if not ranked:
                raise RuntimeError("响应没有有效的候选文档下标")

            # The API normally returns top_n entries. Complete short responses deterministically.
            ranked.extend(item for index, item in enumerate(items) if index not in consumed)
            return RerankResult(items=ranked[:limit], applied=True, model=self.model)
        except Exception as ex:
            logger.warning("Qwen3-Rerank 失败，保留混合召回顺序: %s", ex)
            return RerankResult(items=items[:limit], applied=False, model=self.model, error=str(ex))

    async def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(self.endpoint, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as ex:
            # DashScope puts actionable details such as model-permission and
            # workspace-mismatch errors in the JSON response body.
            detail = response.text.strip().replace("\n", " ")[:1000]
            message = f"HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message) from ex
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Qwen3-Rerank 返回了非 JSON 对象")
        return data

    @staticmethod
    def _document_text(item: Any) -> str:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            return f"标题：{title}\n内容：{content}" if title else content
        return str(item)

    def _attach_score(self, item: Any, score: Any, index: int) -> Any:
        if not isinstance(item, dict):
            return item
        ranked = dict(item)
        try:
            rerank_score = float(score)
        except (TypeError, ValueError):
            rerank_score = 0.0
        if "score" in ranked:
            ranked["retrieval_score"] = ranked["score"]
        ranked["score"] = round(rerank_score, 6)
        ranked["rerank_score"] = round(rerank_score, 6)
        ranked["reranker_model"] = self.model
        ranked["rerank_index"] = index
        return ranked
