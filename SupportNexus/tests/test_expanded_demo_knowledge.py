import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "data" / "demo_docs" / "build_expanded_b2b_saas_knowledge.py"
DATA_PATH = ROOT / "data" / "demo_docs" / "expanded_b2b_saas_knowledge.json"
MANIFEST_PATH = ROOT / "data" / "demo_docs" / "expanded_b2b_saas_manifest.json"


def test_expanded_demo_corpus_has_sufficient_unique_retrieval_content():
    module = runpy.run_path(str(SCRIPT_PATH))
    documents = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert documents == module["build_documents"]()
    assert len(documents) == 220
    assert len({document["title"] for document in documents}) == len(documents)
    assert all(len(document["content"]) >= 350 for document in documents)
    assert manifest["source_document_count"] == 220
    assert manifest["estimated_chunk_count"] >= 360
    assert len(manifest["domains"]) == 12
