import importlib.util
import json
import sqlite3
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_build_indexes.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_indexes", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_input_chunks_prefers_merged_file(tmp_path):
    module = load_module()
    output_dir = tmp_path / "_rag_output"
    output_dir.mkdir()
    chunks = output_dir / "chunks.jsonl"
    merged = output_dir / "chunks_merged_after_ocr.jsonl"
    chunks.write_text("base\n", encoding="utf-8")
    merged.write_text("merged\n", encoding="utf-8")

    assert module.select_input_chunks(output_dir) == merged

    merged.unlink()
    assert module.select_input_chunks(output_dir) == chunks


def test_prepare_chunks_applies_defaults_rejects_invalid_and_dedupes():
    module = load_module()
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_title": "标题",
            "article_no": "第一条",
            "content": "正文",
            "source_file": "/tmp/a.pdf",
        },
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_title": "标题",
            "article_no": "第二条",
            "content": "重复",
            "source_file": "/tmp/a.pdf",
        },
        {"chunk_id": "bad", "doc_id": "d2", "doc_title": "坏", "content": "", "source_file": "/tmp/b.pdf"},
    ]

    valid, invalid, duplicates = module.prepare_chunks(rows)

    assert len(valid) == 1
    assert valid[0]["status"] == "未知"
    assert valid[0]["access_level"] == "internal"
    assert valid[0]["parse_method"] == "text"
    assert "文件标题：标题" in valid[0]["content_for_embedding"]
    assert duplicates == ["c1"]
    assert invalid and invalid[0]["chunk_id"] == "bad"


def test_stable_uuid_from_chunk_id_is_deterministic():
    module = load_module()

    assert module.stable_uuid_from_chunk_id("doc001_article_001") == module.stable_uuid_from_chunk_id(
        "doc001_article_001"
    )
    assert module.stable_uuid_from_chunk_id("doc001_article_001") != module.stable_uuid_from_chunk_id(
        "doc001_article_002"
    )


def test_sqlite_schema_fts_and_insert_roundtrip(tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "doc_title": "艾滋病防治条例",
        "doc_no": None,
        "issuing_body": None,
        "publish_date": None,
        "effective_date": None,
        "expiry_date": None,
        "status": "未知",
        "scope": None,
        "law_level": None,
        "category_path": "传染病防治/艾滋病",
        "category_level_1": "传染病防治",
        "category_level_2": "艾滋病",
        "category_level_3": None,
        "category_level_4": None,
        "chapter": "第一章 总则",
        "section": None,
        "article_no": "第一条",
        "clause_no": None,
        "content": "第一条 为了预防控制艾滋病。",
        "source_file": "/tmp/a.pdf",
        "relative_source_file": "a.pdf",
        "page_start": 1,
        "page_end": 1,
        "access_level": "internal",
        "parse_method": "text",
        "content_for_embedding": "文件标题：艾滋病防治条例\n正文：第一条 为了预防控制艾滋病。",
        "content_length": 15,
    }

    conn = sqlite3.connect(db_path)
    tokenizer = module.create_sqlite_schema(conn)
    module.upsert_sqlite_chunks(conn, [chunk])
    check = module.verify_sqlite(conn, [chunk])
    conn.close()

    assert tokenizer in {"trigram", "unicode61"}
    assert check["law_chunks_count"] == 1
    assert check["fts_query_ok"] is True
    assert check["fts_sample_result_count"] >= 1


def test_build_qdrant_payload_excludes_full_content_and_has_preview():
    module = load_module()
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "doc_title": "标题",
        "doc_no": None,
        "issuing_body": None,
        "publish_date": None,
        "effective_date": None,
        "expiry_date": None,
        "status": "未知",
        "scope": None,
        "law_level": None,
        "category_path": "分类",
        "category_level_1": "分类",
        "category_level_2": None,
        "category_level_3": None,
        "category_level_4": None,
        "chapter": None,
        "section": None,
        "article_no": "第一条",
        "clause_no": None,
        "content": "甲" * 500,
        "source_file": "/tmp/a.pdf",
        "relative_source_file": "a.pdf",
        "page_start": 1,
        "page_end": 1,
        "access_level": "internal",
        "parse_method": "ocr",
    }

    payload = module.build_qdrant_payload(chunk)

    assert "content" not in payload
    assert payload["content_preview"] == "甲" * 300
    assert payload["chunk_id"] == "c1"


def test_verify_qdrant_search_embeds_first_300_chars(monkeypatch):
    module = load_module()
    calls = {}

    def fake_embed_texts(texts, model, ollama_url):
        calls["texts"] = texts
        calls["model"] = model
        calls["ollama_url"] = ollama_url
        return [[0.1, 0.2]]

    class FakeClient:
        def query_points(self, collection_name, query, limit):
            assert collection_name == "law_chunks_local"
            assert query == [0.1, 0.2]
            assert limit == 3
            return type("Response", (), {"points": [object(), object(), object()]})()

    monkeypatch.setattr(module, "embed_texts", fake_embed_texts)

    check = module.verify_qdrant_search(
        FakeClient(),
        "law_chunks_local",
        {"content_for_embedding": "甲" * 350},
        "qwen3-embedding:0.6b",
        "http://localhost:11434",
    )

    assert calls == {
        "texts": ["甲" * 300],
        "model": "qwen3-embedding:0.6b",
        "ollama_url": "http://localhost:11434",
    }
    assert check == {"ok": True, "top_k": 3, "result_count": 3}
