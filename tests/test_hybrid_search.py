import importlib.util
import sqlite3
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "04_hybrid_search.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hybrid_search", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_law_chunks_db(db_path: Path, rows: list[dict], with_fts: bool = False) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE law_chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            doc_title TEXT,
            doc_no TEXT,
            issuing_body TEXT,
            publish_date TEXT,
            effective_date TEXT,
            expiry_date TEXT,
            status TEXT,
            scope TEXT,
            law_level TEXT,
            category_path TEXT,
            category_level_1 TEXT,
            category_level_2 TEXT,
            category_level_3 TEXT,
            category_level_4 TEXT,
            chapter TEXT,
            section TEXT,
            article_no TEXT,
            clause_no TEXT,
            content TEXT,
            source_file TEXT,
            relative_source_file TEXT,
            page_start INTEGER,
            page_end INTEGER,
            access_level TEXT,
            parse_method TEXT,
            content_for_embedding TEXT,
            content_length INTEGER,
            created_at TEXT
        )
        """
    )
    columns = [
        "chunk_id",
        "doc_id",
        "doc_title",
        "doc_no",
        "issuing_body",
        "publish_date",
        "effective_date",
        "expiry_date",
        "status",
        "scope",
        "law_level",
        "category_path",
        "category_level_1",
        "category_level_2",
        "category_level_3",
        "category_level_4",
        "chapter",
        "section",
        "article_no",
        "clause_no",
        "content",
        "source_file",
        "relative_source_file",
        "page_start",
        "page_end",
        "access_level",
        "parse_method",
        "content_for_embedding",
        "content_length",
        "created_at",
    ]
    for row in rows:
        full = {column: None for column in columns}
        full.update(row)
        full.setdefault("access_level", "internal")
        full.setdefault("parse_method", "text")
        full["content_length"] = len(full.get("content") or "")
        conn.execute(
            f"INSERT INTO law_chunks ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})",
            [full[column] for column in columns],
        )
    if with_fts:
        conn.execute(
            """
            CREATE VIRTUAL TABLE law_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                doc_title,
                doc_no,
                issuing_body,
                category_path,
                chapter,
                section,
                article_no,
                content,
                content='law_chunks',
                content_rowid='rowid',
                tokenize='unicode61'
            )
            """
        )
        conn.execute("INSERT INTO law_chunks_fts(law_chunks_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()


def test_normalize_query_and_filters_are_safe():
    module = load_module()

    assert module.normalize_query_for_fts('  "艾滋病": (报告)  ') == "艾滋病 报告"

    where_sql, params = module.build_sql_where_filters(
        {"status": "现行有效", "category": "传染病", "doc_title": "条例", "access_level": "internal"}
    )

    assert "law_chunks.status = ?" in where_sql
    assert "law_chunks.category_path LIKE ?" in where_sql
    assert "law_chunks.doc_title LIKE ?" in where_sql
    assert "law_chunks.access_level = ?" in where_sql
    assert params == ["现行有效", "%传染病%", "%条例%", "internal"]


def test_keyword_search_falls_back_to_like_and_applies_filters(tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    create_law_chunks_db(
        db_path,
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "doc_title": "艾滋病防治条例",
                "article_no": "第三十八条",
                "content": "疫情报告应当遵守传染病报告制度。",
                "category_path": "传染病防治/艾滋病",
                "status": "现行有效",
                "source_file": "/tmp/a.pdf",
                "relative_source_file": "a.pdf",
                "access_level": "internal",
            },
            {
                "chunk_id": "c2",
                "doc_id": "d2",
                "doc_title": "职业病防治法",
                "article_no": "第一条",
                "content": "职业病报告要求。",
                "category_path": "职业卫生",
                "status": "现行有效",
                "source_file": "/tmp/b.pdf",
                "relative_source_file": "b.pdf",
                "access_level": "internal",
            },
        ],
        with_fts=False,
    )

    results = module.keyword_search("疫情报告", db_path, top_k=5, filters={"category": "艾滋病"})

    assert [result["chunk_id"] for result in results] == ["c1"]
    assert results[0]["keyword_rank"] == 1
    assert "keyword_score" in results[0]
    assert results[0]["content"] == "疫情报告应当遵守传染病报告制度。"


def test_keyword_search_falls_back_to_like_when_fts_returns_no_rows(tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    create_law_chunks_db(
        db_path,
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "doc_title": "艾滋病防治条例",
                "article_no": "第三十八条",
                "content": "艾滋病疫情报告应当遵守传染病报告制度。",
                "category_path": "传染病防治/艾滋病",
                "status": "现行有效",
                "source_file": "/tmp/a.pdf",
                "relative_source_file": "a.pdf",
                "access_level": "internal",
            }
        ],
        with_fts=True,
    )

    results = module.keyword_search("艾滋病疫情报告有什么要求", db_path, top_k=5)

    assert [result["chunk_id"] for result in results] == ["c1"]
    assert results[0]["keyword_rank"] == 1


def test_get_chunks_by_ids_skips_missing_and_returns_full_rows(tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    create_law_chunks_db(
        db_path,
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "doc_title": "标题",
                "article_no": "第一条",
                "content": "完整正文",
                "source_file": "/tmp/a.pdf",
            }
        ],
    )

    chunks = module.get_chunks_by_ids(["missing", "c1"], db_path)

    assert list(chunks) == ["c1"]
    assert chunks["c1"]["content"] == "完整正文"
    assert chunks["c1"]["doc_id"] == "d1"


def test_apply_python_filters_uses_exact_and_contains_rules():
    module = load_module()
    rows = [
        {
            "chunk_id": "c1",
            "status": "现行有效",
            "access_level": "internal",
            "category_path": "传染病防治/艾滋病",
            "doc_title": "艾滋病防治条例",
        },
        {
            "chunk_id": "c2",
            "status": "未知",
            "access_level": "internal",
            "category_path": "职业卫生",
            "doc_title": "职业病防治法",
        },
    ]

    filtered = module.apply_python_filters(rows, {"status": "现行有效", "category": "艾滋病", "doc_title": "条例"})

    assert [row["chunk_id"] for row in filtered] == ["c1"]


def test_reciprocal_rank_fusion_merges_sources_and_keeps_single_route_results():
    module = load_module()
    keyword_results = [
        {"chunk_id": "c1", "keyword_rank": 1, "keyword_score": -0.3, "doc_title": "关键词结果"},
        {"chunk_id": "c2", "keyword_rank": 2, "keyword_score": -0.8, "doc_title": "仅关键词"},
    ]
    semantic_results = [
        {"chunk_id": "c3", "semantic_rank": 1, "semantic_score": 0.9, "doc_title": "仅语义"},
        {"chunk_id": "c1", "semantic_rank": 2, "semantic_score": 0.8, "doc_title": "语义结果"},
    ]

    fused = module.reciprocal_rank_fusion(keyword_results, semantic_results, top_k=3, rrf_k=60)

    assert fused[0]["chunk_id"] == "c1"
    assert fused[0]["retrieval_sources"] == ["keyword", "semantic"]
    assert fused[0]["keyword_rank"] == 1
    assert fused[0]["semantic_rank"] == 2
    assert {row["chunk_id"] for row in fused} == {"c1", "c2", "c3"}


def test_semantic_search_hydrates_full_content_and_applies_python_filters(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    create_law_chunks_db(
        db_path,
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "doc_title": "艾滋病防治条例",
                "article_no": "第三十八条",
                "content": "完整疫情报告内容",
                "category_path": "传染病防治/艾滋病",
                "status": "现行有效",
                "source_file": "/tmp/a.pdf",
                "relative_source_file": "a.pdf",
                "access_level": "internal",
            },
            {
                "chunk_id": "c2",
                "doc_id": "d2",
                "doc_title": "职业病防治法",
                "article_no": "第一条",
                "content": "职业病内容",
                "category_path": "职业卫生",
                "status": "现行有效",
                "source_file": "/tmp/b.pdf",
                "relative_source_file": "b.pdf",
                "access_level": "internal",
            },
        ],
    )
    monkeypatch.setattr(module, "SQLITE_DB", db_path)
    monkeypatch.setattr(module, "embed_query", lambda query, model, url: [0.1, 0.2])

    class FakePoint:
        def __init__(self, chunk_id, score):
            self.payload = {"chunk_id": chunk_id}
            self.score = score

    class FakeResponse:
        points = [FakePoint("c1", 0.9), FakePoint("c2", 0.8)]

    class FakeClient:
        def __init__(self, url):
            self.url = url

        def query_points(self, collection_name, query, limit, query_filter=None):
            assert collection_name == "law_chunks_local"
            assert query == [0.1, 0.2]
            assert limit == 10
            return FakeResponse()

    monkeypatch.setattr(module, "QdrantClient", FakeClient)

    results = module.semantic_search(
        "疫情报告",
        qdrant_url="http://localhost:6333",
        collection="law_chunks_local",
        embedding_model="qwen3-embedding:0.6b",
        ollama_url="http://localhost:11434",
        top_k=10,
        filters={"category": "艾滋病"},
    )

    assert [result["chunk_id"] for result in results] == ["c1"]
    assert results[0]["semantic_rank"] == 1
    assert results[0]["semantic_score"] == 0.9
    assert results[0]["content"] == "完整疫情报告内容"


def test_hybrid_search_returns_keyword_results_when_semantic_fails(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    create_law_chunks_db(
        db_path,
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "doc_title": "艾滋病防治条例",
                "article_no": "第三十八条",
                "content": "疫情报告内容",
                "category_path": "传染病防治/艾滋病",
                "status": "现行有效",
                "source_file": "/tmp/a.pdf",
                "relative_source_file": "a.pdf",
                "access_level": "internal",
                "parse_method": "text",
            }
        ],
    )

    monkeypatch.setattr(
        module,
        "keyword_search",
        lambda *args, **kwargs: [
            {
                "chunk_id": "c1",
                "keyword_rank": 1,
                "keyword_score": -0.1,
                "doc_title": "艾滋病防治条例",
            }
        ],
    )
    monkeypatch.setattr(module, "semantic_search", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Qdrant down")))

    results = module.hybrid_search("疫情报告", db_path=db_path, top_k=3)

    assert len(results) == 1
    assert results[0]["rank"] == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["content"] == "疫情报告内容"
    assert results[0]["warnings"] == ["语义检索失败：Qdrant down"]
    assert "content_for_embedding" not in results[0]
    assert "content_length" not in results[0]
    assert "created_at" not in results[0]
