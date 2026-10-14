#!/usr/bin/env python3
"""
构建本地法规 RAG 检索索引。

输入最终 chunks.jsonl，写入 SQLite 主表与 FTS5 全文索引，并通过本地
Ollama embedding 模型写入 Qdrant 向量库。
"""

import argparse
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_DIR = Path(os.environ.get("CDC_LAW_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUTPUT_DIR = ROOT_DIR / "_rag_output"
MERGED_CHUNKS = OUTPUT_DIR / "chunks_merged_after_ocr.jsonl"
BASE_CHUNKS = OUTPUT_DIR / "chunks.jsonl"
SQLITE_DB = OUTPUT_DIR / "law_rag.db"
INDEX_REPORT = OUTPUT_DIR / "index_report.json"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_COLLECTION = "law_chunks_local"

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

REQUIRED_FIELDS = ("chunk_id", "doc_id", "doc_title", "article_no", "content", "source_file")
DEFAULT_FIELDS = {
    "doc_no": None,
    "issuing_body": None,
    "publish_date": None,
    "effective_date": None,
    "expiry_date": None,
    "status": "未知",
    "scope": None,
    "law_level": None,
    "category_path": None,
    "category_level_1": None,
    "category_level_2": None,
    "category_level_3": None,
    "category_level_4": None,
    "chapter": None,
    "section": None,
    "clause_no": None,
    "relative_source_file": None,
    "page_start": None,
    "page_end": None,
    "access_level": "internal",
    "parse_method": "text",
}

SQLITE_COLUMNS = [
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

QDRANT_PAYLOAD_FIELDS = [
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
    "source_file",
    "relative_source_file",
    "page_start",
    "page_end",
    "access_level",
    "parse_method",
]


class IndexBuildError(RuntimeError):
    """Raised for clear, user-facing index build failures."""


def select_input_chunks(output_dir: Path = OUTPUT_DIR) -> Path:
    merged = output_dir / "chunks_merged_after_ocr.jsonl"
    base = output_dir / "chunks.jsonl"
    if merged.exists():
        return merged
    if base.exists():
        return base
    raise FileNotFoundError(f"未找到 chunks 文件：{merged} 或 {base}")


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    total_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_invalid_json": True, "_line_no": line_no, "_error": str(exc)})
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                rows.append({"_invalid_json": True, "_line_no": line_no, "_error": "JSON line is not an object"})
    return rows, total_lines


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_content_for_embedding(chunk: Dict[str, Any]) -> str:
    return f"""
文件标题：{chunk.get("doc_title") or ""}
文号：{chunk.get("doc_no") or ""}
发文机关：{chunk.get("issuing_body") or ""}
法规层级：{chunk.get("law_level") or ""}
分类：{chunk.get("category_path") or ""}
章节：{chunk.get("chapter") or ""}
节：{chunk.get("section") or ""}
条号：{chunk.get("article_no") or ""}
款号：{chunk.get("clause_no") or ""}
效力状态：{chunk.get("status") or ""}
适用范围：{chunk.get("scope") or ""}
正文：
{chunk.get("content") or ""}
""".strip()


def normalize_chunk(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if raw.get("_invalid_json"):
        return None, {
            "line_no": raw.get("_line_no"),
            "chunk_id": None,
            "reason": f"JSON 解析失败: {raw.get('_error')}",
        }

    chunk = dict(DEFAULT_FIELDS)
    chunk.update(raw)

    missing = [field for field in REQUIRED_FIELDS if not normalize_text(chunk.get(field))]
    if missing:
        return None, {
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "reason": "必填字段为空: " + ", ".join(missing),
        }

    for field in [
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
        "access_level",
        "parse_method",
    ]:
        chunk[field] = normalize_text(chunk.get(field))

    chunk["status"] = chunk.get("status") or "未知"
    chunk["access_level"] = chunk.get("access_level") or "internal"
    chunk["parse_method"] = chunk.get("parse_method") or "text"
    chunk["page_start"] = normalize_int(chunk.get("page_start"))
    chunk["page_end"] = normalize_int(chunk.get("page_end"))
    chunk["content_length"] = len(chunk["content"])
    chunk["content_for_embedding"] = build_content_for_embedding(chunk)
    return chunk, None


def prepare_chunks(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    duplicates: List[str] = []
    seen_chunk_ids = set()

    for raw in rows:
        chunk, invalid_info = normalize_chunk(raw)
        if invalid_info is not None:
            invalid.append(invalid_info)
            continue
        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_chunk_ids:
            duplicates.append(chunk_id)
            continue
        seen_chunk_ids.add(chunk_id)
        valid.append(chunk)

    return valid, invalid, duplicates


def stable_uuid_from_chunk_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_sqlite_schema(conn: sqlite3.Connection) -> str:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS law_chunks (
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
    for sql in [
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_doc_id ON law_chunks(doc_id)",
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_doc_title ON law_chunks(doc_title)",
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_status ON law_chunks(status)",
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_category_path ON law_chunks(category_path)",
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_article_no ON law_chunks(article_no)",
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_access_level ON law_chunks(access_level)",
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_publish_date ON law_chunks(publish_date)",
        "CREATE INDEX IF NOT EXISTS idx_law_chunks_effective_date ON law_chunks(effective_date)",
    ]:
        conn.execute(sql)

    conn.execute("DROP TABLE IF EXISTS law_chunks_fts")
    tokenizer = "trigram"
    try:
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
                tokenize='trigram'
            )
            """
        )
    except sqlite3.OperationalError:
        tokenizer = "unicode61"
        conn.execute("DROP TABLE IF EXISTS law_chunks_fts")
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
    conn.commit()
    return tokenizer


def upsert_sqlite_chunks(conn: sqlite3.Connection, chunks: List[Dict[str, Any]]) -> int:
    timestamp = now_iso()
    placeholders = ", ".join(["?"] * len(SQLITE_COLUMNS))
    update_columns = [col for col in SQLITE_COLUMNS if col != "chunk_id"]
    update_clause = ", ".join([f"{col}=excluded.{col}" for col in update_columns])
    sql = f"""
        INSERT INTO law_chunks ({", ".join(SQLITE_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(chunk_id) DO UPDATE SET {update_clause}
    """
    rows = []
    for chunk in chunks:
        row = []
        for col in SQLITE_COLUMNS:
            if col == "created_at":
                row.append(timestamp)
            else:
                row.append(chunk.get(col))
        rows.append(row)
    conn.executemany(sql, rows)
    conn.execute("INSERT INTO law_chunks_fts(law_chunks_fts) VALUES('rebuild')")
    conn.commit()
    return len(chunks)


def fts_escape_query(term: str) -> str:
    term = term.replace('"', '""').strip()
    return f'"{term}"'


def verify_sqlite(conn: sqlite3.Connection, sample_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = conn.execute("SELECT COUNT(*) FROM law_chunks").fetchone()[0]
    terms: List[str] = []
    for chunk in sample_chunks[:5]:
        for key in ("doc_title", "article_no", "category_path"):
            value = chunk.get(key)
            if value:
                terms.append(str(value))
    terms.append("第一条")

    fts_query_ok = False
    sample_result_count = 0
    for term in terms:
        try:
            query = fts_escape_query(term)
            sample_result_count = conn.execute(
                "SELECT COUNT(*) FROM law_chunks_fts WHERE law_chunks_fts MATCH ?",
                (query,),
            ).fetchone()[0]
            fts_query_ok = True
            if sample_result_count > 0:
                break
        except sqlite3.Error:
            continue

    return {
        "law_chunks_count": count,
        "fts_query_ok": fts_query_ok,
        "fts_sample_result_count": sample_result_count,
    }


def check_dependencies() -> None:
    missing = []
    for module_name in ("qdrant_client", "requests", "tqdm"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise IndexBuildError("请先安装依赖：\npip install qdrant-client requests tqdm")


def embed_texts(texts: List[str], model: str, ollama_url: str) -> List[List[float]]:
    import requests

    url = ollama_url.rstrip("/") + "/api/embed"
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(url, json={"model": model, "input": texts}, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise IndexBuildError(
            "无法连接 Ollama，请确认 ollama serve 已启动，并已 pull 对应 embedding 模型。"
        ) from exc

    embeddings = data.get("embeddings")
    if embeddings is None and "embedding" in data:
        embeddings = [data["embedding"]]
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise IndexBuildError("Ollama embedding 返回格式异常，未找到与输入数量一致的 embeddings。")
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise IndexBuildError("Ollama embedding 返回了空向量。")
    return embeddings


def import_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        return QdrantClient, Distance, PointStruct, VectorParams
    except ImportError as exc:
        raise IndexBuildError("请先安装依赖：\npip install qdrant-client requests tqdm") from exc


def build_qdrant_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    payload = {field: chunk.get(field) for field in QDRANT_PAYLOAD_FIELDS}
    payload["content_preview"] = (chunk.get("content") or "")[:300]
    return payload


def qdrant_collection_exists(client: Any, collection: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection))
    try:
        client.get_collection(collection)
        return True
    except Exception:
        return False


def ensure_qdrant_collection(
    client: Any,
    collection: str,
    dimension: int,
    rebuild: bool,
    Distance: Any,
    VectorParams: Any,
) -> None:
    exists = qdrant_collection_exists(client, collection)
    if exists and rebuild:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        return

    info = client.get_collection(collection)
    vector_size = None
    vectors = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(vectors, "vectors", None)
    if hasattr(vectors, "size"):
        vector_size = vectors.size
    if vector_size is not None and int(vector_size) != int(dimension):
        raise IndexBuildError(
            f"Qdrant collection 向量维度为 {vector_size}，当前 embedding 维度为 {dimension}；请使用 --rebuild 重建。"
        )


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def upsert_qdrant_chunks(
    chunks: List[Dict[str, Any]],
    model: str,
    ollama_url: str,
    qdrant_url: str,
    collection: str,
    batch_size: int,
    rebuild: bool,
) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    if not chunks:
        return 0, 0, {"collection_exists": False, "points_count": 0}, {
            "ok": False,
            "top_k": 3,
            "result_count": 0,
        }

    QdrantClient, Distance, PointStruct, VectorParams = import_qdrant()
    client = QdrantClient(url=qdrant_url)

    first_embedding = embed_texts([chunks[0]["content_for_embedding"]], model, ollama_url)[0]
    dimension = len(first_embedding)
    ensure_qdrant_collection(client, collection, dimension, rebuild, Distance, VectorParams)

    upserted = 0
    first_batch = True
    from tqdm import tqdm

    for chunk_batch in tqdm(list(batched(chunks, batch_size)), desc="写入 Qdrant"):
        if first_batch:
            texts = [chunk["content_for_embedding"] for chunk in chunk_batch[1:]]
            embeddings = [first_embedding] + (embed_texts(texts, model, ollama_url) if texts else [])
            first_batch = False
        else:
            embeddings = embed_texts(
                [chunk["content_for_embedding"] for chunk in chunk_batch], model, ollama_url
            )
        points = [
            PointStruct(
                id=stable_uuid_from_chunk_id(chunk["chunk_id"]),
                vector=embedding,
                payload=build_qdrant_payload(chunk),
            )
            for chunk, embedding in zip(chunk_batch, embeddings)
        ]
        client.upsert(collection_name=collection, points=points)
        upserted += len(points)

    qdrant_check = verify_qdrant(client, collection)
    search_check = verify_qdrant_search(client, collection, chunks[0], model, ollama_url)
    return upserted, dimension, qdrant_check, search_check


def verify_qdrant(client: Any, collection: str) -> Dict[str, Any]:
    exists = qdrant_collection_exists(client, collection)
    points_count = 0
    if exists:
        info = client.get_collection(collection)
        points_count = getattr(info, "points_count", None)
        if points_count is None:
            points_count = client.count(collection_name=collection, exact=True).count
    return {"collection_exists": exists, "points_count": int(points_count or 0)}


def verify_qdrant_search(
    client: Any, collection: str, sample_chunk: Dict[str, Any], model: str, ollama_url: str
) -> Dict[str, Any]:
    top_k = 3
    try:
        sample_text = (sample_chunk.get("content_for_embedding") or "")[:300]
        if not sample_text:
            return {"ok": False, "top_k": top_k, "result_count": 0}
        sample_embedding = embed_texts([sample_text], model, ollama_url)[0]
        if hasattr(client, "search"):
            results = client.search(collection_name=collection, query_vector=sample_embedding, limit=top_k)
        else:
            response = client.query_points(collection_name=collection, query=sample_embedding, limit=top_k)
            results = getattr(response, "points", response)
        return {"ok": True, "top_k": top_k, "result_count": len(results)}
    except Exception:
        return {"ok": False, "top_k": top_k, "result_count": 0}


def write_report(report: Dict[str, Any]) -> None:
    INDEX_REPORT.parent.mkdir(parents=True, exist_ok=True)
    INDEX_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_initial_report(
    input_chunks: Optional[Path],
    args: argparse.Namespace,
    total_input_lines: int = 0,
) -> Dict[str, Any]:
    return {
        "root_dir": str(ROOT_DIR),
        "input_chunks": str(input_chunks) if input_chunks else None,
        "sqlite_db": str(SQLITE_DB),
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.collection,
        "embedding_model": args.embedding_model,
        "sqlite_fts_tokenizer": None,
        "total_input_lines": total_input_lines,
        "valid_chunk_count": 0,
        "invalid_chunk_count": 0,
        "duplicate_chunk_count": 0,
        "sqlite_inserted_count": 0,
        "qdrant_upserted_count": 0,
        "embedding_dimension": None,
        "rebuild": bool(args.rebuild),
        "errors": [],
        "invalid_chunks": [],
        "duplicate_chunks": [],
        "sqlite_check": {
            "law_chunks_count": 0,
            "fts_query_ok": False,
            "fts_sample_result_count": 0,
        },
        "qdrant_check": {
            "collection_exists": False,
            "points_count": 0,
        },
        "qdrant_search_check": {
            "ok": False,
            "top_k": 3,
            "result_count": 0,
        },
    }


def build_indexes(args: argparse.Namespace) -> Dict[str, Any]:
    input_chunks = select_input_chunks(OUTPUT_DIR)
    rows, total_input_lines = load_jsonl(input_chunks)
    report = build_initial_report(input_chunks, args, total_input_lines)

    valid_chunks, invalid_chunks, duplicate_chunks = prepare_chunks(rows)
    report["valid_chunk_count"] = len(valid_chunks)
    report["invalid_chunk_count"] = len(invalid_chunks)
    report["duplicate_chunk_count"] = len(duplicate_chunks)
    report["invalid_chunks"] = invalid_chunks
    report["duplicate_chunks"] = duplicate_chunks

    if args.rebuild and SQLITE_DB.exists():
        SQLITE_DB.unlink()

    conn = sqlite3.connect(SQLITE_DB)
    try:
        tokenizer = create_sqlite_schema(conn)
        report["sqlite_fts_tokenizer"] = tokenizer
        report["sqlite_inserted_count"] = upsert_sqlite_chunks(conn, valid_chunks)
        report["sqlite_check"] = verify_sqlite(conn, valid_chunks)
    finally:
        conn.close()

    check_dependencies()
    upserted, dimension, qdrant_check, qdrant_search_check = upsert_qdrant_chunks(
        chunks=valid_chunks,
        model=args.embedding_model,
        ollama_url=args.ollama_url,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )
    report["qdrant_upserted_count"] = upserted
    report["embedding_dimension"] = dimension
    report["qdrant_check"] = qdrant_check
    report["qdrant_search_check"] = qdrant_search_check
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建法规 RAG 本地检索索引")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def print_summary(report: Dict[str, Any]) -> None:
    print("\n索引构建完成")
    print(f"根目录：{ROOT_DIR}")
    print(f"输入 chunks：{report.get('input_chunks')}")
    print(f"有效 chunk 数：{report.get('valid_chunk_count')}")
    print(f"SQLite 数据库：{SQLITE_DB}")
    print(f"Qdrant collection：{report.get('qdrant_collection')}")
    print(f"Embedding 模型：{report.get('embedding_model')}")
    print(f"向量维度：{report.get('embedding_dimension')}")
    print(f"报告：{INDEX_REPORT}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        print("错误：--batch-size 必须大于 0")
        sys.exit(1)

    input_chunks: Optional[Path] = None
    try:
        input_chunks = select_input_chunks(OUTPUT_DIR)
        report = build_indexes(args)
    except Exception as exc:
        report = build_initial_report(input_chunks, args)
        report["errors"].append(str(exc))
        write_report(report)
        print(f"错误：{exc}")
        sys.exit(1)

    write_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
