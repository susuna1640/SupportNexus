from collections import Counter

from mcp.knowledge_base import KnowledgeBase


class FakeCollection:
    def __init__(self):
        self.metadata = {"description": "test knowledge base"}
        self.rows = [
            ("refund", "退款将在三个工作日原路到账。", {"title": "退款到账", "chunk_index": 0}),
            ("policy", "退款申请需要符合售后政策。", {"title": "退款政策", "chunk_index": 0}),
            ("security", "账户安全需要开启双重验证。", {"title": "账户安全", "chunk_index": 0}),
        ]

    def count(self):
        return len(self.rows)

    def get(self, include, where=None, limit=None, offset=None):
        rows = self.rows
        if where is not None:
            rows = [row for row in rows if row[2].get("title") == where.get("title")]
        start = offset or 0
        rows = rows[start:start + limit] if limit is not None else rows[start:]
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [row[2] for row in rows],
        }

    def delete(self, ids):
        ids = set(ids)
        self.rows = [row for row in self.rows if row[0] not in ids]

    def modify(self, metadata):
        self.metadata = metadata


def make_knowledge_base():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = FakeCollection()
    kb._bm25_doc_count = -1
    kb._bm25_documents = []
    kb._bm25_term_frequencies = []
    kb._bm25_document_frequencies = Counter()
    kb._bm25_avg_doc_len = 0.0
    return kb


def test_bm25_ranks_exact_chinese_keyword_matches():
    kb = make_knowledge_base()

    results = kb.search_bm25("退款多久到账", top_k=2)

    assert results[0]["id"] == "refund"
    assert results[0]["retrieval"] == "bm25"
    assert results[0]["score"] > results[1]["score"]


def test_rrf_promotes_a_chunk_returned_by_both_retrievers():
    kb = make_knowledge_base()
    dense = [
        {"id": "dense-only", "content": "语义相近", "score": 0.9},
        {"id": "both", "content": "退款到账时间", "score": 0.8},
    ]
    sparse = [
        {"id": "both", "content": "退款到账时间", "score": 5.0},
        {"id": "sparse-only", "content": "退款政策", "score": 4.0},
    ]

    results = kb._fuse_hybrid_results(dense, sparse, top_k=3)

    assert results[0]["id"] == "both"
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1
    assert results[0]["retrieval"] == "hybrid_rrf"


def test_knowledge_chunk_management_lists_and_deletes_only_requested_title():
    kb = make_knowledge_base()

    chunks = kb.list_chunks()
    deleted = kb.delete_chunks_by_title("退款政策")

    assert [chunk["id"] for chunk in chunks] == ["refund", "policy", "security"]
    assert deleted == 1
    assert kb.doc_count == 2
    assert [chunk["title"] for chunk in kb.list_chunks()] == ["退款到账", "账户安全"]


def test_clear_chunks_only_removes_the_knowledge_base_collection_contents():
    kb = make_knowledge_base()

    deleted = kb.clear_chunks()

    assert deleted == 3
    assert kb.doc_count == 0
    assert kb._collection.metadata[KnowledgeBase.USER_CLEARED_KEY] is True


def test_seed_default_documents_requires_an_empty_knowledge_base():
    kb = make_knowledge_base()

    try:
        kb.seed_default_documents()
    except ValueError as ex:
        assert "知识库非空" in str(ex)
    else:
        raise AssertionError("非空知识库不应重复导入默认文档")
