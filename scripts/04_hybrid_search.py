#!/usr/bin/env python3
"""
法规 RAG 本地混合检索层。

本脚本只读取上一阶段生成的 SQLite 与 Qdrant 索引，执行关键词检索、
向量检索、RRF 融合、元数据过滤和结果打印。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import requests

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue
except ImportError:  # pragma: no cover - exercised through dependency checks in real use.
    QdrantClient = None
    FieldCondition = None
    Filter = None
    MatchValue = None


ROOT_DIR = Path(os.environ.get("CDC_LAW_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUTPUT_DIR = ROOT_DIR / "_rag_output"
SQLITE_DB = OUTPUT_DIR / "law_rag.db"
SEARCH_REPORT = OUTPUT_DIR / "search_test_report.json"

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "law_chunks_local"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

CHUNK_COLUMNS = [
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

RESULT_COLUMNS = [
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
]

FINAL_RESULT_FIELDS = [
    "rank",
    "chunk_id",
    "score",
    "retrieval_sources",
    "keyword_rank",
    "semantic_rank",
    "keyword_score",
    "semantic_score",
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
    "chapter",
    "section",
    "article_no",
    "clause_no",
    "content",
    "page_start",
    "page_end",
    "source_file",
    "relative_source_file",
    "access_level",
    "parse_method",
    "warnings",
]

LIKE_FIELDS = [
    "doc_title",
    "doc_no",
    "issuing_body",
    "category_path",
    "chapter",
    "section",
    "article_no",
    "content",
]

MAX_LIKE_TERMS = 40


class HybridSearchError(RuntimeError):
    """Raised for clear user-facing retrieval failures."""


def check_dependencies() -> None:
    missing = []
    for module_name in ("qdrant_client", "requests", "tqdm"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise HybridSearchError("请先安装依赖：\npip install qdrant-client requests tqdm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="法规 RAG 本地混合检索")
    parser.add_argument("--query", help="用户查询")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--top-k", type=int, default=8, help="最终返回数量")
    parser.add_argument("--keyword-top-k", type=int, default=30, help="SQLite 关键词召回数量")
    parser.add_argument("--semantic-top-k", type=int, default=30, help="Qdrant 向量召回数量")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 参数")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--status", help="按法规状态精确过滤")
    parser.add_argument("--category", help="按 category_path 模糊过滤")
    parser.add_argument("--doc-title", help="按 doc_title 模糊过滤")
    parser.add_argument("--access-level", help="按 access_level 精确过滤")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    parser.add_argument("--save-report", action="store_true", help="保存检索测试报告")
    return parser.parse_args()


def normalize_query_for_fts(query: str) -> str:
    if not query.strip():
        return query
    text = re.sub(r'[\x00-\x1f"\'`^~*?:()\[\]{}<>|\\!+\-/]', " ", query)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if not filters:
        return {}
    return {key: value for key, value in filters.items() if value is not None and str(value).strip()}


def build_sql_where_filters(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    active = compact_filters(filters)
    clauses: list[str] = []
    params: list[Any] = []
    if active.get("status"):
        clauses.append("law_chunks.status = ?")
        params.append(active["status"])
    if active.get("category"):
        clauses.append("law_chunks.category_path LIKE ?")
        params.append(f"%{active['category']}%")
    if active.get("doc_title"):
        clauses.append("law_chunks.doc_title LIKE ?")
        params.append(f"%{active['doc_title']}%")
    if active.get("access_level"):
        clauses.append("law_chunks.access_level = ?")
        params.append(active["access_level"])
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def keyword_search(
    query: str,
    db_path: Path,
    top_k: int = 30,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_query_for_fts(query)
    if not normalized.strip():
        return []

    where_sql, filter_params = build_sql_where_filters(filters)
    select_cols = ", ".join([f"law_chunks.{column}" for column in RESULT_COLUMNS])
    conn = connect_db(db_path)
    try:
        try:
            sql = f"""
                SELECT {select_cols}, bm25(law_chunks_fts) AS keyword_score
                FROM law_chunks_fts
                JOIN law_chunks ON law_chunks_fts.rowid = law_chunks.rowid
                WHERE law_chunks_fts MATCH ?{where_sql}
                ORDER BY keyword_score ASC
                LIMIT ?
            """
            rows = conn.execute(sql, [normalized, *filter_params, top_k]).fetchall()
            if not rows:
                rows = keyword_search_like(conn, normalized, top_k, filters)
        except sqlite3.Error:
            rows = keyword_search_like(conn, normalized, top_k, filters)
    finally:
        conn.close()

    results = [row_to_dict(row) for row in rows]
    for rank, result in enumerate(results, start=1):
        result["keyword_rank"] = rank
        result["keyword_score"] = float(result.get("keyword_score") or 0.0)
    return results


def extract_like_terms(query: str) -> list[str]:
    normalized = normalize_query_for_fts(query)
    if not normalized.strip():
        return []
    terms: list[str] = []
    seen = set()

    def add(term: str) -> None:
        term = term.strip()
        if len(term) < 2 or term in seen:
            return
        seen.add(term)
        terms.append(term)

    for token in normalized.split():
        add(token)
        cjk_runs = re.findall(r"[\u4e00-\u9fff]+", token)
        for run in cjk_runs:
            if len(run) <= 4:
                add(run)
                continue
            for size in (3, 2):
                for start in range(0, len(run) - size + 1):
                    add(run[start : start + size])
                    if len(terms) >= MAX_LIKE_TERMS:
                        return terms
    return terms[:MAX_LIKE_TERMS]


def keyword_search_like(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[sqlite3.Row]:
    terms = extract_like_terms(query) or [query]
    term_clauses = []
    params: list[Any] = []
    score_parts = []
    score_params: list[Any] = []
    for term in terms:
        field_clauses = []
        for field in LIKE_FIELDS:
            field_clauses.append(f"law_chunks.{field} LIKE ?")
            params.append(f"%{term}%")
            score_parts.append(f"CASE WHEN law_chunks.{field} LIKE ? THEN 1 ELSE 0 END")
            score_params.append(f"%{term}%")
        term_clauses.append("(" + " OR ".join(field_clauses) + ")")

    where_sql, filter_params = build_sql_where_filters(filters)
    select_cols = ", ".join([f"law_chunks.{column}" for column in RESULT_COLUMNS])
    score_expr = " + ".join(score_parts) if score_parts else "0"
    sql = f"""
        SELECT {select_cols}, -1.0 * ({score_expr}) AS keyword_score
        FROM law_chunks
        WHERE ({" OR ".join(term_clauses)}){where_sql}
        ORDER BY keyword_score ASC, law_chunks.chunk_id ASC
        LIMIT ?
    """
    return conn.execute(sql, [*score_params, *params, *filter_params, top_k]).fetchall()


def embed_query(query: str, model: str, ollama_url: str) -> list[float]:
    url = ollama_url.rstrip("/") + "/api/embed"
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(url, json={"model": model, "input": query}, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise HybridSearchError("无法连接 Ollama，请确认 ollama serve 已启动，并已 pull 对应 embedding 模型。") from exc

    embeddings = data.get("embeddings")
    if embeddings is None and "embedding" in data:
        embeddings = [data["embedding"]]
    if not isinstance(embeddings, list) or not embeddings:
        raise HybridSearchError("Ollama embedding 返回格式异常，未找到 embeddings。")
    embedding = embeddings[0]
    if not isinstance(embedding, list) or not embedding:
        raise HybridSearchError("Ollama embedding 返回了空向量。")
    return embedding


def build_qdrant_filter(filters: dict[str, Any] | None):
    if Filter is None or FieldCondition is None or MatchValue is None:
        return None
    active = compact_filters(filters)
    conditions = []
    if active.get("status"):
        conditions.append(FieldCondition(key="status", match=MatchValue(value=active["status"])))
    if active.get("access_level"):
        conditions.append(FieldCondition(key="access_level", match=MatchValue(value=active["access_level"])))
    if not conditions:
        return None
    return Filter(must=conditions)


def semantic_search(
    query: str,
    qdrant_url: str,
    collection: str,
    embedding_model: str,
    ollama_url: str,
    top_k: int = 30,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if QdrantClient is None:
        raise HybridSearchError("请先安装依赖：\npip install qdrant-client requests tqdm")

    embedding = embed_query(query, embedding_model, ollama_url)
    qdrant_filter = build_qdrant_filter(filters)
    try:
        client = QdrantClient(url=qdrant_url)
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=collection,
                query=embedding,
                limit=top_k,
                query_filter=qdrant_filter,
            )
            points = getattr(response, "points", response)
        else:
            points = client.search(
                collection_name=collection,
                query_vector=embedding,
                limit=top_k,
                query_filter=qdrant_filter,
            )
    except Exception as exc:
        raise HybridSearchError("无法连接 Qdrant，请确认 qdrant 服务已启动。") from exc

    ranked_points = []
    chunk_ids = []
    for rank, point in enumerate(points, start=1):
        payload = getattr(point, "payload", None) or {}
        chunk_id = payload.get("chunk_id")
        if not chunk_id:
            continue
        score = getattr(point, "score", None)
        ranked_points.append((chunk_id, rank, float(score or 0.0), payload))
        chunk_ids.append(chunk_id)

    chunks_by_id = get_chunks_by_ids(chunk_ids, SQLITE_DB)
    results = []
    for chunk_id, rank, score, payload in ranked_points:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            chunk = {"chunk_id": chunk_id, **payload}
        result = dict(chunk)
        result["semantic_rank"] = rank
        result["semantic_score"] = score
        results.append(result)
    return apply_python_filters(results, filters)


def get_chunks_by_ids(chunk_ids: list[str], db_path: Path) -> dict[str, dict[str, Any]]:
    ordered_ids = []
    seen = set()
    for chunk_id in chunk_ids:
        if chunk_id and chunk_id not in seen:
            ordered_ids.append(chunk_id)
            seen.add(chunk_id)
    if not ordered_ids:
        return {}

    placeholders = ", ".join(["?"] * len(ordered_ids))
    conn = connect_db(db_path)
    try:
        rows = conn.execute(
            f"SELECT {', '.join(CHUNK_COLUMNS)} FROM law_chunks WHERE chunk_id IN ({placeholders})",
            ordered_ids,
        ).fetchall()
    finally:
        conn.close()

    by_id = {row["chunk_id"]: row_to_dict(row) for row in rows}
    return {chunk_id: by_id[chunk_id] for chunk_id in ordered_ids if chunk_id in by_id}


def apply_python_filters(results: list[dict[str, Any]], filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    active = compact_filters(filters)
    if not active:
        return results

    filtered = []
    for result in results:
        if active.get("status") and result.get("status") != active["status"]:
            continue
        if active.get("access_level") and result.get("access_level") != active["access_level"]:
            continue
        if active.get("category") and active["category"] not in (result.get("category_path") or ""):
            continue
        if active.get("doc_title") and active["doc_title"] not in (result.get("doc_title") or ""):
            continue
        filtered.append(result)
    return filtered


def reciprocal_rank_fusion(
    keyword_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    top_k: int = 8,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def ensure(chunk_id: str, source_result: dict[str, Any]) -> dict[str, Any]:
        if chunk_id not in merged:
            merged[chunk_id] = {
                **source_result,
                "chunk_id": chunk_id,
                "score": 0.0,
                "keyword_rank": None,
                "semantic_rank": None,
                "keyword_score": None,
                "semantic_score": None,
                "retrieval_sources": [],
            }
        else:
            for key, value in source_result.items():
                if key not in merged[chunk_id] or merged[chunk_id].get(key) is None:
                    merged[chunk_id][key] = value
        return merged[chunk_id]

    for fallback_rank, result in enumerate(keyword_results, start=1):
        chunk_id = result.get("chunk_id")
        if not chunk_id:
            continue
        rank = int(result.get("keyword_rank") or fallback_rank)
        item = ensure(chunk_id, result)
        item["score"] += 1.0 / (rrf_k + rank)
        item["keyword_rank"] = rank
        item["keyword_score"] = result.get("keyword_score")
        if "keyword" not in item["retrieval_sources"]:
            item["retrieval_sources"].append("keyword")

    for fallback_rank, result in enumerate(semantic_results, start=1):
        chunk_id = result.get("chunk_id")
        if not chunk_id:
            continue
        rank = int(result.get("semantic_rank") or fallback_rank)
        item = ensure(chunk_id, result)
        item["score"] += 1.0 / (rrf_k + rank)
        item["semantic_rank"] = rank
        item["semantic_score"] = result.get("semantic_score")
        if "semantic" not in item["retrieval_sources"]:
            item["retrieval_sources"].append("semantic")

    return sorted(merged.values(), key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]


def hybrid_search(
    query: str,
    db_path: Path = SQLITE_DB,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    top_k: int = 8,
    keyword_top_k: int = 30,
    semantic_top_k: int = 30,
    rrf_k: int = 60,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise HybridSearchError("查询为空")

    warnings = []
    try:
        keyword_results = keyword_search(query, db_path, keyword_top_k, filters)
    except Exception as exc:
        keyword_results = []
        warnings.append(f"关键词检索失败：{exc}")

    try:
        semantic_results = semantic_search(
            query=query,
            qdrant_url=qdrant_url,
            collection=collection,
            embedding_model=embedding_model,
            ollama_url=ollama_url,
            top_k=semantic_top_k,
            filters=filters,
        )
    except Exception as exc:
        semantic_results = []
        warnings.append(f"语义检索失败：{exc}")

    if not keyword_results and not semantic_results:
        return []

    fused = reciprocal_rank_fusion(keyword_results, semantic_results, top_k=top_k, rrf_k=rrf_k)
    chunks_by_id = get_chunks_by_ids([item["chunk_id"] for item in fused], db_path)
    final_results = []
    for item in fused:
        row = dict(chunks_by_id.get(item["chunk_id"], {}))
        row.update(item)
        if warnings:
            row["warnings"] = warnings
        final_results.append(row)

    final_results = apply_python_filters(final_results, filters)
    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank
    return [clean_final_result(result) for result in final_results]


def clean_final_result(result: dict[str, Any]) -> dict[str, Any]:
    cleaned = {field: result.get(field) for field in FINAL_RESULT_FIELDS if field in result}
    if "warnings" in cleaned and not cleaned["warnings"]:
        cleaned.pop("warnings")
    return cleaned


def truncate_content(content: str, max_chars: int = 800) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "\n……"


def format_page_range(result: dict[str, Any]) -> str:
    page_start = result.get("page_start")
    page_end = result.get("page_end")
    if page_start and page_end and page_start != page_end:
        return f"{page_start}-{page_end}"
    if page_start:
        return str(page_start)
    if page_end:
        return str(page_end)
    return ""


def print_results(results: list[dict[str, Any]], query: str) -> None:
    print(f"查询：{query}")
    print(f"返回结果：{len(results)} 条")
    if not results:
        print("无检索结果")
        return
    for index, result in enumerate(results, start=1):
        print(f"==================== {index} ====================")
        print(f"score: {float(result.get('score') or 0.0):.4f}")
        print(f"sources: {', '.join(result.get('retrieval_sources') or [])}")
        print(f"keyword_rank: {result.get('keyword_rank')}")
        print(f"semantic_rank: {result.get('semantic_rank')}")
        print(f"文件：{result.get('doc_title') or ''}")
        print(f"条号：{result.get('article_no') or ''}")
        print(f"章节：{result.get('chapter') or ''}")
        print(f"页码：{format_page_range(result)}")
        print(f"状态：{result.get('status') or ''}")
        print(f"分类：{result.get('category_path') or ''}")
        print(f"来源：{result.get('relative_source_file') or result.get('source_file') or ''}")
        if result.get("warnings"):
            print(f"warning: {'; '.join(result['warnings'])}")
        print("正文：")
        print(truncate_content(result.get("content") or ""))


def save_search_report(
    query: str,
    filters: dict[str, Any],
    top_k: int,
    keyword_top_k: int,
    semantic_top_k: int,
    results: list[dict[str, Any]],
) -> Path:
    report = {
        "query": query,
        "filters": {
            "status": filters.get("status"),
            "category": filters.get("category"),
            "doc_title": filters.get("doc_title"),
            "access_level": filters.get("access_level"),
        },
        "top_k": top_k,
        "keyword_top_k": keyword_top_k,
        "semantic_top_k": semantic_top_k,
        "result_count": len(results),
        "results": [
            {
                "rank": result.get("rank"),
                "chunk_id": result.get("chunk_id"),
                "doc_title": result.get("doc_title"),
                "article_no": result.get("article_no"),
                "score": result.get("score"),
                "retrieval_sources": result.get("retrieval_sources"),
                "content_preview": (result.get("content") or "")[:300],
            }
            for result in results
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return SEARCH_REPORT


def qdrant_collection_exists(client: Any, collection: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection))
    try:
        client.get_collection(collection)
        return True
    except Exception:
        return False


def preflight_check(
    db_path: Path = SQLITE_DB,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[str]:
    check_dependencies()
    if not db_path.exists():
        raise HybridSearchError(f"找不到 SQLite 数据库：{db_path}\n请先运行 scripts/03_build_indexes.py")

    conn = connect_db(db_path)
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            ).fetchall()
        }
    finally:
        conn.close()
    if "law_chunks" not in tables:
        raise HybridSearchError("SQLite 数据库中找不到 law_chunks 表，请先运行 scripts/03_build_indexes.py")
    if "law_chunks_fts" not in tables:
        raise HybridSearchError("SQLite 数据库中找不到 law_chunks_fts 表，请先运行 scripts/03_build_indexes.py")

    warnings = []
    try:
        embed_query("测试", embedding_model, ollama_url)
    except Exception as exc:
        warnings.append(str(exc))

    if QdrantClient is None:
        warnings.append("请先安装依赖：\npip install qdrant-client requests tqdm")
    else:
        try:
            client = QdrantClient(url=qdrant_url)
            if not qdrant_collection_exists(client, collection):
                warnings.append(f"Qdrant collection 不存在：{collection}")
        except Exception:
            warnings.append("无法连接 Qdrant，请确认 qdrant 服务已启动。")
    return warnings


def build_filters_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": args.status,
        "category": args.category,
        "doc_title": args.doc_title,
        "access_level": args.access_level,
    }


def run_single_query(args: argparse.Namespace, query: str, preflight_warnings: list[str]) -> list[dict[str, Any]]:
    filters = build_filters_from_args(args)
    results = hybrid_search(
        query=query,
        db_path=SQLITE_DB,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        top_k=args.top_k,
        keyword_top_k=args.keyword_top_k,
        semantic_top_k=args.semantic_top_k,
        rrf_k=args.rrf_k,
        filters=filters,
    )
    if preflight_warnings:
        for result in results:
            result.setdefault("warnings", [])
            for warning in preflight_warnings:
                if warning not in result["warnings"]:
                    result["warnings"].append(warning)
    if args.save_report:
        save_search_report(query, filters, args.top_k, args.keyword_top_k, args.semantic_top_k, results)
    return results


def print_json_output(query: str, results: list[dict[str, Any]]) -> None:
    print(json.dumps({"query": query, "result_count": len(results), "results": results}, ensure_ascii=False, indent=2))


def interactive_loop(args: argparse.Namespace, preflight_warnings: list[str]) -> None:
    print("请输入问题，输入 exit 退出：")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break
        results = run_single_query(args, query, preflight_warnings)
        if args.json:
            print_json_output(query, results)
        else:
            print_results(results, query)


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.keyword_top_k < 1 or args.semantic_top_k < 1 or args.rrf_k < 1:
        print("错误：--top-k、--keyword-top-k、--semantic-top-k、--rrf-k 必须大于 0", file=sys.stderr)
        sys.exit(1)
    if not args.query and not args.interactive:
        print("错误：请提供 --query，或使用 --interactive", file=sys.stderr)
        sys.exit(1)

    try:
        preflight_warnings = preflight_check(
            db_path=SQLITE_DB,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            embedding_model=args.embedding_model,
            ollama_url=args.ollama_url,
        )
    except HybridSearchError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)

    for warning in preflight_warnings:
        print(f"warning: {warning}", file=sys.stderr)

    try:
        if args.interactive:
            interactive_loop(args, preflight_warnings)
            return
        results = run_single_query(args, args.query, preflight_warnings)
        if args.json:
            print_json_output(args.query, results)
        else:
            print_results(results, args.query)
    except HybridSearchError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
