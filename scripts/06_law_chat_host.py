#!/usr/bin/env python3
"""
Ollama 法规 Agent Host。

本脚本把本地 qwen3.6 对话、Ollama tool calling、SQLite/Qdrant 混合检索、
HTTP API 与命令行交互串起来。它不重建索引，也不修改原始数据。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from html import escape, unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover - dependency path is covered by preflight messaging.
    QdrantClient = None


ROOT_DIR = Path(os.environ.get("CDC_LAW_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUTPUT_DIR = ROOT_DIR / "_rag_output"
SQLITE_DB = OUTPUT_DIR / "law_rag.db"
AGENT_LOG_DIR = OUTPUT_DIR / "agent_logs"
OFFICIAL_UPDATES_PENDING_DIR = OUTPUT_DIR / "official_updates_pending"
HYBRID_SEARCH_SCRIPT = ROOT_DIR / "scripts" / "04_hybrid_search.py"
AGENTIC_RAG_SCRIPT = ROOT_DIR / "scripts" / "05_agentic_rag.py"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.6:27b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "law_chunks_local"

DEPENDENCY_MESSAGE = "请先安装依赖：\npip install requests qdrant-client fastapi uvicorn pydantic tqdm"

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

hybrid_search = None
_hybrid_module = None

LAW_KEYWORDS = [
    "法规",
    "法律",
    "条例",
    "办法",
    "规定",
    "制度",
    "规范",
    "职责",
    "义务",
    "处罚",
    "罚款",
    "报告",
    "备案",
    "审批",
    "许可",
    "监督",
    "检查",
    "执法",
    "行政",
    "传染病",
    "职业病",
    "艾滋病",
    "食品安全",
    "卫生",
    "医疗机构",
    "疾控",
    "预防",
    "控制",
    "应当",
    "不得",
    "可以",
    "能否",
    "是否",
]

OFFICIAL_WEB_HOSTS = {
    "flk.npc.gov.cn",
    "wb.flk.npc.gov.cn",
    "www.nhc.gov.cn",
    "www.ndcpa.gov.cn",
    "www.chinacdc.cn",
    "www.gov.cn",
    "xzfg.moj.gov.cn",
}

OFFICIAL_WEB_HOST_SUFFIXES = {
    ".gov.cn",
    ".chinacdc.cn",
}

OFFICIAL_READER_PREFIX = "https://r.jina.ai/"
OFFICIAL_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
}

OFFICIAL_INDEX_SOURCES = [
    {
        "source": "国家卫生健康委员会",
        "base_url": "https://www.nhc.gov.cn",
        "urls": [
            "https://www.nhc.gov.cn/",
            "https://www.nhc.gov.cn/fzs/new_index.shtml",
            "https://www.nhc.gov.cn/zwgkzt/pzcfg/list.shtml",
            "https://www.nhc.gov.cn/zwgkzt/pzcfg1/list.shtml",
            "https://www.nhc.gov.cn/zhengce/",
            "https://www.nhc.gov.cn/xcs/zhengcwj/list.shtml",
            "https://www.nhc.gov.cn/wjw/gfxwj/list.shtml",
            "https://www.nhc.gov.cn/wjw/zcjd/list.shtml",
            "https://www.nhc.gov.cn/wjw/gggs/list.shtml",
            "https://www.nhc.gov.cn/ylyjs/gzdt/list.shtml",
            "https://www.nhc.gov.cn/ylyjs/zcwj/list.shtml",
            "https://www.nhc.gov.cn/jkj/",
            "https://www.nhc.gov.cn/jws/",
        ],
    },
    {
        "source": "国家疾病预防控制局",
        "base_url": "https://www.ndcpa.gov.cn",
        "urls": [
            "https://www.ndcpa.gov.cn/jbkzzx/c100011/common/list.html",
            "https://www.ndcpa.gov.cn/jbkzzx/c100012/common/list.html",
            "https://www.ndcpa.gov.cn/jbkzzx/c100013/common/list.html",
            "https://www.ndcpa.gov.cn/jbkzzx/c100016/common/list.html",
            "https://www.ndcpa.gov.cn/jbkzzx/c1000372/common/list.html",
            "https://www.ndcpa.gov.cn/jbkzzx/c100041/common/list.html",
            "https://www.ndcpa.gov.cn/jbkzzx/ZCWJ/common/list.html",
        ],
    },
    {
        "source": "中国疾病预防控制中心",
        "base_url": "https://www.chinacdc.cn",
        "urls": [
            "https://www.chinacdc.cn/",
            "https://www.chinacdc.cn/jkyj/",
            "https://www.chinacdc.cn/jkyj/yyyjk2/jswj13949/",
            "https://www.chinacdc.cn/jkyj/fsws/fgbz/fg/gfxwj/",
            "https://www.chinacdc.cn/jkzt/",
            "https://ivdc.chinacdc.cn/cnic/zyzx/syssc/",
        ],
    },
    {
        "source": "中国政府网",
        "base_url": "https://www.gov.cn",
        "urls": ["https://www.gov.cn/zhengce/"],
    },
    {
        "source": "司法部国家行政法规库",
        "base_url": "https://xzfg.moj.gov.cn",
        "urls": ["https://xzfg.moj.gov.cn/"],
    },
]

OFFICIAL_WEB_TRIGGER_KEYWORDS = [
    "最新",
    "官网",
    "官方",
    "国家规定",
    "国家卫健委",
    "卫健委",
    "国家卫生健康委",
    "国家疾控局",
    "疾控局",
    "中国疾控",
    "现行",
    "目前",
    "新规",
    "发布",
    "政策文件",
    "通知",
    "202",
]

SQL_COLUMNS = [
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
    "content_length",
    "created_at",
]


class LawChatHostError(RuntimeError):
    """Raised for clear user-facing host failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 Ollama 法规 Agent Host")
    parser.add_argument("--interactive", action="store_true", help="命令行聊天模式")
    parser.add_argument("--serve", action="store_true", help="HTTP 服务模式")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--keyword-top-k", type=int, default=30)
    parser.add_argument("--semantic-top-k", type=int, default=30)
    parser.add_argument("--max-agent-steps", type=int, default=6)
    parser.add_argument("--status")
    parser.add_argument("--category")
    parser.add_argument("--doc-title")
    parser.add_argument("--access-level")
    parser.add_argument("--save-log", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--no-web", action="store_true", help="禁用官方联网检索兜底")
    parser.add_argument("--web-timeout", type=int, default=12, help="官方联网检索超时时间，单位秒")
    parser.add_argument("--no-official-download", action="store_true", help="禁用官方候选文件下载")
    parser.add_argument("--official-updates-dir", type=Path, default=OFFICIAL_UPDATES_PENDING_DIR)
    parser.add_argument("--download-timeout", type=int, default=20, help="官方候选文件下载超时时间，单位秒")
    parser.add_argument("--max-download-bytes", type=int, default=50_000_000, help="单个官方候选文件最大下载字节数")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    interactive = bool(args.interactive or not args.serve)
    return {
        "root_dir": ROOT_DIR,
        "output_dir": OUTPUT_DIR,
        "sqlite_db": SQLITE_DB,
        "interactive": interactive,
        "serve": bool(args.serve),
        "host": args.host,
        "port": int(args.port),
        "ollama_url": args.ollama_url,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "qdrant_url": args.qdrant_url,
        "collection": args.collection,
        "top_k": int(args.top_k),
        "keyword_top_k": int(args.keyword_top_k),
        "semantic_top_k": int(args.semantic_top_k),
        "max_agent_steps": int(args.max_agent_steps),
        "num_predict": 1024,
        "ollama_timeout": 300,
        "status": args.status,
        "category": args.category,
        "doc_title": args.doc_title,
        "access_level": args.access_level,
        "save_log": bool(args.save_log),
        "debug": bool(args.debug),
        "no_tools": bool(args.no_tools),
        "web_enabled": not bool(getattr(args, "no_web", False)),
        "web_timeout": int(getattr(args, "web_timeout", 12)),
        "official_updates_dir": Path(getattr(args, "official_updates_dir", OFFICIAL_UPDATES_PENDING_DIR) or OFFICIAL_UPDATES_PENDING_DIR),
        "official_download_enabled": not bool(getattr(args, "no_official_download", False)),
        "download_timeout": int(getattr(args, "download_timeout", 20)),
        "max_download_bytes": int(getattr(args, "max_download_bytes", 50_000_000)),
    }


def check_dependencies(require_server: bool = False) -> None:
    required = ["requests", "qdrant_client", "tqdm"]
    if require_server:
        required.extend(["fastapi", "uvicorn", "pydantic"])
    missing = []
    for module_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise LawChatHostError(DEPENDENCY_MESSAGE)


def load_module_from_path(path: Path, name: str):
    if not path.exists():
        raise LawChatHostError(f"找不到脚本：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_hybrid_search_module():
    global _hybrid_module, hybrid_search
    if _hybrid_module is None:
        _hybrid_module = load_module_from_path(HYBRID_SEARCH_SCRIPT, "hybrid_search_module")
        hybrid_search = _hybrid_module.hybrid_search
    return _hybrid_module


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def preflight_check(config: dict[str, Any]) -> None:
    check_dependencies(require_server=bool(config.get("serve")))
    db_path = Path(config["sqlite_db"])
    if not db_path.exists():
        raise LawChatHostError("找不到 SQLite 数据库，请先运行：\npython scripts/03_build_indexes.py")
    if not HYBRID_SEARCH_SCRIPT.exists():
        raise LawChatHostError(f"找不到混合检索脚本：{HYBRID_SEARCH_SCRIPT}")

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
        raise LawChatHostError("SQLite 数据库中找不到 law_chunks 表，请先运行：\npython scripts/03_build_indexes.py")

    tags = get_ollama_model_names(config["ollama_url"])
    if config["llm_model"] not in tags:
        raise LawChatHostError(
            f"模型 {config['llm_model']} 不存在，请先执行：\nollama pull {config['llm_model']}"
        )
    if config["embedding_model"] not in tags:
        raise LawChatHostError(
            f"模型 {config['embedding_model']} 不存在，请先执行：\nollama pull {config['embedding_model']}"
        )

    if QdrantClient is None:
        raise LawChatHostError(DEPENDENCY_MESSAGE)
    try:
        client = QdrantClient(url=config["qdrant_url"], check_compatibility=False)
        exists = client.collection_exists(config["collection"]) if hasattr(client, "collection_exists") else True
        if not exists:
            raise LawChatHostError(f"Qdrant collection 不存在：{config['collection']}")
    except LawChatHostError:
        raise
    except Exception as exc:
        raise LawChatHostError("无法连接 Qdrant，请确认 qdrant 服务已启动。") from exc


def get_ollama_model_names(ollama_url: str) -> set[str]:
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(ollama_url.rstrip("/") + "/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise LawChatHostError("无法连接 Ollama，请确认已执行：\nollama serve") from exc
    names = set()
    for model in data.get("models", []):
        name = model.get("name") or model.get("model")
        if name:
            names.add(name)
    return names


def get_system_prompt() -> str:
    return """你是一个本地单位法律法规知识库 Agent。
你可以通过工具访问本地法规数据库。
你不能直接访问数据库，必须通过工具调用。
凡是用户问题涉及法规、条例、办法、制度、规定、职责、处罚、报告义务、审批、备案、许可、监督检查、卫生监督、传染病、职业病、艾滋病、食品安全、医疗机构等内容，必须先调用 search_laws。
如果 search_laws 返回的是 preview，而你需要做关键结论，必须调用 read_chunk 读取完整条文。
如果涉及当前是否适用，必须调用 check_validity。
当本地库没有明确依据、用户要求最新/官网/官方/国家规定，或问题明显超出本地库时，再调用 official_web_search 查询官方白名单网站。
最终回答必须只基于工具返回的证据；本地库证据与官网联网证据要区分说明。
不要凭模型常识编造法规依据。
如果没有检索到明确依据，必须回答“未检索到明确依据”。
回答风格要自然、简洁，先直接回答用户问题，再列依据；不要为了套模板重复空泛段落。
引用本地库证据时必须给出文件名、条号、页码、chunk_id。
引用官网联网证据时必须给出来源网站、标题、发布日期或发布时间（如有）、URL。
如果 official_web_search 返回 fetched_via=reader_fallback 且包含 snippet/content_preview，说明已通过只读通道读取到官网正文，不要仅因 fetch_warnings 就回答无法访问。
证据不足时直接说明缺口，不要补写不存在的依据。"""


def build_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_laws",
                "description": "检索本地法律法规、条例、制度、办法、规范性文件数据库。适合回答法规依据、职责、处罚、报告、审批、许可、监督检查等问题。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索问题或关键词"},
                        "top_k": {"type": "integer", "description": "返回结果数量"},
                        "status": {"type": "string", "description": "法规状态过滤，例如 现行有效"},
                        "category": {"type": "string", "description": "分类过滤，例如 传染病防治/艾滋病"},
                        "doc_title": {"type": "string", "description": "文件标题过滤"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": "按 chunk_id 读取完整法规条文内容。",
                "parameters": {
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string", "description": "chunk_id"}},
                    "required": ["chunk_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_validity",
                "description": "检查法规文件状态、发布日期、生效日期、废止日期。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "doc_title": {"type": "string"},
                        "chunk_id": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_categories",
                "description": "列出本地法规库中的分类路径。",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "最多返回数量"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "official_web_search",
                "description": "在官方白名单网站中检索最新或本地库未覆盖的法律法规、国家卫健委、国家疾控局、中国疾控、国务院政策文件等信息。只返回官方来源链接。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要检索的法规、政策或卫生防疫问题"},
                        "limit": {"type": "integer", "description": "最多返回数量，默认 5"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def ollama_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, config: dict[str, Any], stream: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config["llm_model"],
        "messages": messages,
        "stream": bool(stream),
        "options": {"temperature": 0.1, "num_predict": int(config.get("num_predict") or 1024)},
    }
    if tools:
        body["tools"] = tools
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(
            config["ollama_url"].rstrip("/") + "/api/chat",
            json=body,
            timeout=int(config.get("ollama_timeout") or 300),
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if len(detail) > 1000:
                detail = detail[:1000] + "..."
            raise LawChatHostError(f"Ollama 返回 {response.status_code}：{detail}") from exc
        return response.json()
    except LawChatHostError:
        raise
    except Exception as exc:
        raise LawChatHostError("无法连接 Ollama 或模型调用失败，请确认 ollama serve 已启动且模型已安装。") from exc


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(message_content_to_text(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") in {"text", "input_text"} and "text" in content:
            return message_content_to_text(content.get("text"))
        for key in ("text", "content", "input_text"):
            if key in content:
                return message_content_to_text(content.get(key))
        return ""
    return str(content)


def normalize_host_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message_content_to_text(message.get("content")).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def extract_latest_user_message(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = message_content_to_text(message.get("content")).strip()
        if content:
            return content, normalize_host_history(messages[:index])
    return "", []


def is_law_related_question(question: str) -> bool:
    text = message_content_to_text(question)
    return any(keyword in text for keyword in LAW_KEYWORDS)


def build_filters(config: dict[str, Any], status=None, category=None, doc_title=None, access_level=None) -> dict[str, Any]:
    return {
        "status": status if status is not None else config.get("status"),
        "category": category if category is not None else config.get("category"),
        "doc_title": doc_title if doc_title is not None else config.get("doc_title"),
        "access_level": access_level if access_level is not None else config.get("access_level"),
    }


def search_laws(
    query: str,
    top_k: int = 8,
    status=None,
    category=None,
    doc_title=None,
    access_level=None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or build_config(parse_args())
    global hybrid_search
    if hybrid_search is None:
        load_hybrid_search_module()
    filters = build_filters(config, status=status, category=category, doc_title=doc_title, access_level=access_level)
    try:
        rows = hybrid_search(
            query=query,
            db_path=config["sqlite_db"],
            qdrant_url=config["qdrant_url"],
            collection=config["collection"],
            embedding_model=config["embedding_model"],
            ollama_url=config["ollama_url"],
            top_k=top_k or config["top_k"],
            keyword_top_k=config["keyword_top_k"],
            semantic_top_k=config["semantic_top_k"],
            filters=filters,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "results": []}

    results = []
    for row in rows:
        content = row.get("content") or ""
        results.append(
            {
                "chunk_id": row.get("chunk_id"),
                "doc_title": row.get("doc_title"),
                "doc_no": row.get("doc_no"),
                "article_no": row.get("article_no"),
                "chapter": row.get("chapter"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "status": row.get("status"),
                "category_path": row.get("category_path"),
                "score": row.get("score"),
                "content_preview": content[:500],
            }
        )
    return {"ok": True, "query": query, "result_count": len(results), "results": results}


def is_allowed_official_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in OFFICIAL_WEB_HOSTS or any(host.endswith(suffix) for suffix in OFFICIAL_WEB_HOST_SUFFIXES)


def extract_official_urls(text: str) -> list[str]:
    urls = []
    for match in re.finditer(r"https?://[^\s<>'\"\]）)】》]+", text or ""):
        url = match.group(0).rstrip(".,;:!?，。；：！？、")
        if is_allowed_official_url(url) and url not in urls:
            urls.append(url)
    return urls


def source_name_for_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host in {"flk.npc.gov.cn", "wb.flk.npc.gov.cn"}:
        return "国家法律法规数据库"
    if host == "www.nhc.gov.cn":
        return "国家卫生健康委员会"
    if host == "www.ndcpa.gov.cn":
        return "国家疾病预防控制局"
    if host.endswith(".chinacdc.cn") or host == "www.chinacdc.cn":
        return "中国疾病预防控制中心"
    if host == "www.gov.cn":
        return "中国政府网"
    if host == "xzfg.moj.gov.cn":
        return "司法部国家行政法规库"
    if host.endswith(".gov.cn"):
        return "中国政府官方网站"
    return "官方来源"


def clean_html_text(text: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", text or "")
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def clean_markdown_text(text: str) -> str:
    text = re.sub(r"(?m)^Title:\s*", "", text or "")
    text = re.sub(r"(?m)^URL Source:\s*\S+\s*", "", text)
    text = re.sub(r"(?m)^Markdown Content:\s*", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#*_`>]+", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def extract_html_title(html: str) -> str:
    for pattern in (
        r"(?is)<h1[^>]*>(.*?)</h1>",
        r"(?is)<meta[^>]+name=[\"']ArticleTitle[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"(?is)<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+name=[\"']ArticleTitle[\"']",
        r"(?is)<title[^>]*>(.*?)</title>",
    ):
        match = re.search(pattern, html or "")
        if match:
            title = clean_html_text(match.group(1))
            if title:
                return title
    return ""


def extract_attachment_urls(text: str, base_url: str) -> list[str]:
    attachment_urls = []
    html_hrefs = re.findall(r"(?is)<a\b[^>]*href=[\"']([^\"']+)[\"']", text or "")
    markdown_hrefs = re.findall(r"\[[^\]]+\]\((https?://[^)]+|/[^)]+)\)", text or "")
    for href in [*html_hrefs, *markdown_hrefs]:
        candidate = urljoin(base_url, unescape(href).strip())
        ext = extension_from_url_or_content_type(candidate)
        if ext in {".pdf", ".doc", ".docx", ".wps"} and is_allowed_official_url(candidate):
            if candidate not in attachment_urls:
                attachment_urls.append(candidate)
    return attachment_urls


def extract_date(text: str) -> str:
    match = re.search(r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})", text or "")
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def query_terms(query: str) -> list[str]:
    terms = [term for term in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query or "") if len(term) >= 2]
    extra = []
    for word in ["传染病", "艾滋病", "职业病", "卫健委", "疾控局", "报告", "规范", "办法", "条例", "法律", "通知"]:
        if word in (query or ""):
            extra.append(word)
    return list(dict.fromkeys(terms + extra))


def score_official_result(query: str, title: str, snippet: str) -> int:
    haystack = f"{title} {snippet}"
    compact_query = re.sub(r"\s+", "", query or "")
    compact_haystack = re.sub(r"\s+", "", haystack)
    score = 0
    if compact_query and compact_query in compact_haystack:
        score += 20
    for term in query_terms(query):
        if term in haystack:
            score += 3
    return score


def normalize_official_result(result: dict[str, Any], query: str) -> dict[str, Any] | None:
    url = result.get("url") or ""
    if not url or not is_allowed_official_url(url):
        return None
    title = clean_html_text(result.get("title") or "")
    if not title:
        return None
    snippet = clean_html_text(result.get("snippet") or title)
    date = extract_date(result.get("date") or result.get("publish") or snippet)
    normalized = {
        "source": result.get("source") or "",
        "title": title,
        "url": url,
        "date": date,
        "snippet": snippet[:500],
        "score": score_official_result(query, title, snippet),
    }
    if result.get("download_urls"):
        normalized["download_urls"] = result["download_urls"]
    if result.get("flk_id"):
        normalized["flk_id"] = result["flk_id"]
    return normalized


def request_official_text(url: str, timeout: int) -> tuple[str, dict[str, Any], list[str]]:
    warnings = []
    candidates = [url]
    parsed = urlparse(url)
    if parsed.scheme == "https":
        candidates.append("http://" + url[len("https://") :])

    for candidate in candidates:
        for trust_env in (True, False):
            session = requests.Session()
            session.trust_env = trust_env
            try:
                response = session.get(candidate, headers=OFFICIAL_REQUEST_HEADERS, timeout=timeout)
                response.raise_for_status()
                return response.text, {"fetched_via": "official_direct", "fetch_url": candidate}, warnings
            except Exception as exc:
                mode = "环境代理" if trust_env else "直连"
                warnings.append(f"{mode}访问 {candidate} 失败：{exc}")

    raise LawChatHostError("；".join(warnings[-3:]) or "官网页面抓取失败")


def request_reader_content(url: str, timeout: int) -> tuple[str, dict[str, Any], list[str]]:
    reader_url = OFFICIAL_READER_PREFIX + url
    session = requests.Session()
    session.trust_env = True
    try:
        response = session.get(
            reader_url,
            headers={**OFFICIAL_REQUEST_HEADERS, "Accept": "application/json"},
            timeout=max(timeout, 20),
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise LawChatHostError(f"阅读器兜底抓取失败：{exc}") from exc

    payload = data.get("data") if isinstance(data, dict) else {}
    payload = payload if isinstance(payload, dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    content = payload.get("content") or payload.get("description") or ""
    title = payload.get("title") or metadata.get("ArticleTitle") or ""
    date = metadata.get("PubDate") or metadata.get("pubDate") or metadata.get("date") or ""
    if not content and response.text:
        content = response.text
    return (
        content,
        {
            "fetched_via": "reader_fallback",
            "fetch_url": reader_url,
            "title": title,
            "date": date,
            "metadata": metadata,
        },
        [f"官网原站不可达时已启用只读阅读器兜底：{reader_url}"],
    )


def fetch_official_page_content(
    url: str,
    timeout: int,
    reader_fallback_enabled: bool = True,
) -> tuple[str, dict[str, Any], list[str]]:
    try:
        text, metadata, warnings = request_official_text(url, timeout)
        return text, metadata, warnings
    except Exception as exc:
        warnings = [f"官网原站抓取失败：{exc}"]
    if not reader_fallback_enabled:
        raise LawChatHostError(warnings[0])
    try:
        text, metadata, reader_warnings = request_reader_content(url, timeout)
        return text, metadata, [*warnings, *reader_warnings]
    except Exception as exc:
        raise LawChatHostError(f"{warnings[0]}；{exc}") from exc


def official_result_from_direct_url(
    url: str,
    query: str,
    timeout: int,
    reader_fallback_enabled: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    source = source_name_for_url(url)
    try:
        text, metadata, warnings = fetch_official_page_content(url, timeout, reader_fallback_enabled)
        fetched_via = metadata.get("fetched_via") or "official_direct"
        is_html = "<html" in text[:500].lower() or "<body" in text[:1000].lower()
        title = metadata.get("title") or (extract_html_title(text) if is_html else "")
        if not title:
            title_match = re.search(r"(?m)^Title:\s*(.+)$", text or "")
            title = clean_markdown_text(title_match.group(1)) if title_match else ""
        title = title or source
        content_text = clean_html_text(text) if is_html else clean_markdown_text(text)
        date = extract_date(metadata.get("date") or "") or extract_date(content_text)
        download_urls = extract_attachment_urls(text, url)
        item = normalize_official_result(
            {
                "source": source,
                "title": title,
                "url": url,
                "date": date,
                "snippet": content_text or title,
                "download_urls": download_urls,
            },
            query,
        )
        if item:
            item["score"] = max(int(item.get("score") or 0), 100)
            item["fetched_via"] = fetched_via
            item["content_preview"] = (content_text or title)[:1500]
            if metadata.get("fetch_url"):
                item["fetch_url"] = metadata["fetch_url"]
            if warnings:
                item["fetch_warnings"] = warnings[-3:]
        return item, warnings
    except Exception as exc:
        fallback_title = Path(urlparse(url).path).name or source
        item = normalize_official_result(
            {
                "source": source,
                "title": fallback_title,
                "url": url,
                "snippet": f"已识别为官方链接，但当前网络无法抓取正文：{exc}",
            },
            query,
        )
        if item:
            item["score"] = 80
            item["access_status"] = "fetch_failed"
            item["fetch_warnings"] = [str(exc)]
        return item, [str(exc)]


def search_flk_npc_api(session: requests.Session, query: str, limit: int, timeout: int) -> list[dict[str, Any]]:
    params = {
        "type": "flfg",
        "searchType": "title;vague",
        "keyword": query,
        "sortTr": "f_bbrq_s;desc",
        "sort": "true",
        "page": "1",
        "size": str(max(1, min(limit, 10))),
        "_": str(int(time.time() * 1000)),
    }
    response = session.get(
        "https://flk.npc.gov.cn/api/",
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    rows = ((data.get("result") or {}).get("data") or []) if isinstance(data, dict) else []
    results = []
    for row in rows:
        url = row.get("url")
        if not url and row.get("id"):
            url = "https://flk.npc.gov.cn/detail2.html?" + str(row["id"])
        item = normalize_official_result(
            {
                "source": "国家法律法规数据库",
                "title": row.get("title"),
                "url": url,
                "date": row.get("publish") or row.get("expiry"),
                "snippet": "；".join(
                    part
                    for part in [
                        row.get("type"),
                        row.get("office"),
                        row.get("status"),
                        row.get("publish"),
                    ]
                    if part
                ),
            },
            query,
        )
        if item:
            results.append(item)
    return results


def parse_official_links(html: str, base_url: str, source: str, query: str, limit: int) -> list[dict[str, Any]]:
    results = []
    pattern = re.compile(r"(?is)<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>(.{0,200})")
    for match in pattern.finditer(html or ""):
        href, body, tail = match.groups()
        title = clean_html_text(body)
        if len(title) < 4:
            continue
        url = urljoin(base_url, unescape(href).strip())
        if not is_allowed_official_url(url):
            continue
        tail_text = clean_html_text(tail)
        snippet = clean_html_text(f"{title} {tail_text}")[:500]
        item = normalize_official_result(
            {
                "source": source,
                "title": title,
                "url": url,
                "date": extract_date(tail_text) or extract_date(title),
                "snippet": snippet,
            },
            query,
        )
        if item:
            results.append(item)
    results.sort(key=lambda item: (item.get("score", 0), item.get("date") or ""), reverse=True)
    return results[:limit]


def official_web_search(query: str, limit: int = 5, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    if not config.get("web_enabled", True):
        return {"ok": False, "query": query, "result_count": 0, "results": [], "warnings": ["官方联网检索已禁用"]}
    limit = max(1, min(int(limit or 5), 10))
    timeout = int(config.get("web_timeout") or 12)
    reader_fallback_enabled = bool(config.get("reader_fallback_enabled", True))
    session = requests.Session()
    session.trust_env = True
    warnings = []
    results: list[dict[str, Any]] = []

    for url in extract_official_urls(query):
        if len(results) >= limit:
            break
        item, direct_warnings = official_result_from_direct_url(url, query, timeout, reader_fallback_enabled)
        warnings.extend(direct_warnings)
        if item:
            results.append(item)

    if len(results) >= limit:
        deduped = {item["url"]: item for item in results if item.get("url")}
        ranked = sorted(deduped.values(), key=lambda item: (item.get("score", 0), item.get("date") or ""), reverse=True)
        return {
            "ok": bool(ranked),
            "query": query,
            "result_count": len(ranked),
            "results": ranked[:limit],
            "warnings": warnings,
            "searched_sources": ["用户提供的官方直链"],
        }

    try:
        results.extend(search_flk_npc_api(session, query, limit, timeout))
    except Exception as exc:
        warnings.append(f"国家法律法规数据库检索失败：{exc}")

    for source in OFFICIAL_INDEX_SOURCES:
        if len(results) >= limit:
            break
        for url in source["urls"]:
            if len(results) >= limit:
                break
            try:
                response = session.get(url, headers=OFFICIAL_REQUEST_HEADERS, timeout=timeout)
                response.raise_for_status()
                results.extend(
                    parse_official_links(
                        response.text,
                        source["base_url"],
                        source["source"],
                        query,
                        limit,
                    )
                )
            except Exception as exc:
                warnings.append(f"{source['source']}检索失败：{exc}")

    deduped = {}
    for item in results:
        url = item.get("url")
        if url and url not in deduped:
            deduped[url] = item
    ranked = sorted(deduped.values(), key=lambda item: (item.get("score", 0), item.get("date") or ""), reverse=True)
    ranked = ranked[:limit]
    return {
        "ok": bool(ranked),
        "query": query,
        "result_count": len(ranked),
        "results": ranked,
        "warnings": warnings,
        "searched_sources": ["国家法律法规数据库", *[source["source"] for source in OFFICIAL_INDEX_SOURCES]],
    }


def read_chunk(chunk_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or build_config(parse_args())
    conn = connect_db(config["sqlite_db"])
    try:
        row = conn.execute(
            f"SELECT {', '.join(SQL_COLUMNS)} FROM law_chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"ok": False, "error": f"未找到 chunk_id：{chunk_id}"}
    return {"ok": True, "chunk": row_to_dict(row)}


def check_validity(
    doc_id: str | None = None,
    doc_title: str | None = None,
    chunk_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or build_config(parse_args())
    clauses = []
    params = []
    if chunk_id:
        clauses.append("chunk_id = ?")
        params.append(chunk_id)
    if doc_id:
        clauses.append("doc_id = ?")
        params.append(doc_id)
    if doc_title:
        clauses.append("doc_title LIKE ?")
        params.append(f"%{doc_title}%")
    if not clauses:
        return {"ok": False, "error": "请提供 doc_id、doc_title 或 chunk_id"}

    conn = connect_db(config["sqlite_db"])
    try:
        row = conn.execute(
            """
            SELECT doc_id, doc_title, doc_no, issuing_body, publish_date, effective_date,
                   expiry_date, status, law_level, category_path
            FROM law_chunks
            WHERE """
            + " AND ".join(clauses)
            + " LIMIT 1",
            params,
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"ok": False, "error": "未找到对应法规文件"}
    data = row_to_dict(row)
    return {"ok": True, **data}


def list_categories(limit: int = 100, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or build_config(parse_args())
    conn = connect_db(config["sqlite_db"])
    try:
        rows = conn.execute(
            """
            SELECT category_path, COUNT(*) AS chunk_count
            FROM law_chunks
            WHERE category_path IS NOT NULL AND TRIM(category_path) != ''
            GROUP BY category_path
            ORDER BY chunk_count DESC, category_path ASC
            LIMIT ?
            """,
            (int(limit or 100),),
        ).fetchall()
    finally:
        conn.close()
    return {"ok": True, "categories": [row_to_dict(row) for row in rows]}


def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    message = response.get("message") or {}
    raw_calls = message.get("tool_calls") or response.get("tool_calls") or []
    calls = []
    for call in raw_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        item = {"name": name, "arguments": normalize_tool_arguments(function.get("arguments"))}
        if call.get("id"):
            item["id"] = call.get("id")
        calls.append(item)
    return calls


def execute_tool_call(tool_call: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    name = tool_call.get("name")
    args = dict(tool_call.get("arguments") or {})
    try:
        if name == "search_laws":
            result = search_laws(
                query=args.get("query", ""),
                top_k=int(args.get("top_k") or config["top_k"]),
                status=args.get("status"),
                category=args.get("category"),
                doc_title=args.get("doc_title"),
                access_level=args.get("access_level"),
                config=config,
            )
        elif name == "read_chunk":
            result = read_chunk(chunk_id=args.get("chunk_id", ""), config=config)
        elif name == "check_validity":
            result = check_validity(
                doc_id=args.get("doc_id"),
                doc_title=args.get("doc_title"),
                chunk_id=args.get("chunk_id"),
                config=config,
            )
        elif name == "list_categories":
            result = list_categories(limit=int(args.get("limit") or 100), config=config)
        elif name == "official_web_search":
            result = official_web_search(
                query=args.get("query", ""),
                limit=int(args.get("limit") or 5),
                config=config,
            )
        else:
            result = {"ok": False, "error": f"未知工具：{name}"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    return {"name": name, "arguments": args, "result": result}


def trim_chat_history(chat_history: list[dict[str, Any]], max_turn_messages: int = 20) -> list[dict[str, Any]]:
    plain = [message for message in chat_history if message.get("role") in {"user", "assistant"}]
    return plain[-max_turn_messages:]


def initial_messages(user_message: str, chat_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": get_system_prompt()}, *trim_chat_history(chat_history), {"role": "user", "content": user_message}]


def collect_used_chunks_from_tool_result(tool_result: dict[str, Any], used_chunks: dict[str, dict[str, Any]]) -> None:
    result = tool_result.get("result") if "result" in tool_result else tool_result
    if not isinstance(result, dict):
        return
    if isinstance(result.get("results"), list):
        for item in result["results"]:
            chunk_id = item.get("chunk_id")
            if chunk_id:
                used_chunks[chunk_id] = item
    chunk = result.get("chunk")
    if isinstance(chunk, dict) and chunk.get("chunk_id"):
        used_chunks[chunk["chunk_id"]] = chunk


def collect_used_web_sources_from_tool_result(tool_result: dict[str, Any], used_web_sources: dict[str, dict[str, Any]]) -> None:
    result = tool_result.get("result") if "result" in tool_result else tool_result
    if not isinstance(result, dict):
        return
    for item in result.get("results") or []:
        if isinstance(item, dict) and item.get("url"):
            used_web_sources[item["url"]] = item


def latest_tool_result(tool_history: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for entry in reversed(tool_history):
        if entry.get("name") == name:
            result = entry.get("result")
            return result if isinstance(result, dict) else None
    return None


def parse_date_value(value: Any) -> str:
    if not value:
        return ""
    match = re.search(r"(20\d{2}|19\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})", str(value))
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def best_local_date(item: dict[str, Any]) -> str:
    for key in ("effective_date", "publish_date", "created_at"):
        parsed = parse_date_value(item.get(key))
        if parsed:
            return parsed
    return ""


def best_official_date(item: dict[str, Any]) -> str:
    for key in ("date", "publish", "publish_date", "effective_date"):
        parsed = parse_date_value(item.get(key))
        if parsed:
            return parsed
    return ""


def summarize_local_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item.get("chunk_id"),
        "doc_title": item.get("doc_title"),
        "status": item.get("status"),
        "publish_date": item.get("publish_date"),
        "effective_date": item.get("effective_date"),
        "date": best_local_date(item),
    }


def summarize_official_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": item.get("source"),
        "title": item.get("title"),
        "url": item.get("url"),
        "date": best_official_date(item),
        "snippet": item.get("snippet"),
    }


def compare_local_and_official(local_result: dict[str, Any] | None, web_result: dict[str, Any] | None) -> dict[str, Any]:
    local_items = [summarize_local_item(item) for item in (local_result or {}).get("results") or []]
    official_items = [summarize_official_item(item) for item in (web_result or {}).get("results") or []]
    if not official_items:
        return {
            "status": "no_official_result",
            "reason": "未检索到官网候选依据。",
            "local_items": local_items,
            "official_items": official_items,
        }
    if not local_items:
        return {
            "status": "local_missing",
            "reason": "本地库未检索到对应依据，官网结果作为待核对候选。",
            "local_items": local_items,
            "official_items": official_items,
        }

    local_dates = [item["date"] for item in local_items if item.get("date")]
    official_dates = [item["date"] for item in official_items if item.get("date")]
    if not local_dates or not official_dates:
        return {
            "status": "date_unknown",
            "reason": "本地或官网缺少可比较日期，无法谨慎判断新旧。",
            "local_items": local_items,
            "official_items": official_items,
        }

    newest_local = max(local_dates)
    newest_official = max(official_dates)
    if newest_official > newest_local:
        return {
            "status": "official_newer",
            "reason": f"官网日期晚于本地日期：官网 {newest_official}，本地 {newest_local}。",
            "local_items": local_items,
            "official_items": official_items,
        }
    return {
        "status": "possibly_same",
        "reason": f"官网日期未晚于本地日期：官网 {newest_official}，本地 {newest_local}。",
        "local_items": local_items,
        "official_items": official_items,
    }


def should_download_official_candidates(comparison: dict[str, Any] | None, config: dict[str, Any]) -> bool:
    if not config.get("official_download_enabled", True):
        return False
    status = (comparison or {}).get("status")
    return status in {"official_newer", "local_missing", "date_unknown"}


def safe_filename(text: str, max_length: int = 80) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", text or "").strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned or "official_update")[:max_length]


def extension_from_url_or_content_type(url: str, content_type: str = "") -> str:
    path = urlparse(url).path.lower()
    for ext in (".pdf", ".docx", ".doc", ".wps"):
        if path.endswith(ext):
            return ext
    content_type = (content_type or "").lower()
    if "pdf" in content_type:
        return ".pdf"
    if "wordprocessingml" in content_type:
        return ".docx"
    if "msword" in content_type:
        return ".doc"
    return ".html"


def subdir_for_extension(ext: str) -> str:
    if ext == ".pdf":
        return "pdf"
    if ext in {".doc", ".docx", ".wps"}:
        return "word"
    return "html"


def discover_download_urls(item: dict[str, Any], session: requests.Session, timeout: int) -> list[str]:
    urls = [url for url in item.get("download_urls") or [] if isinstance(url, str)]
    if urls:
        return [url for url in urls if is_allowed_official_url(url)]
    url = item.get("url") or ""
    if not url:
        return []
    if extension_from_url_or_content_type(url) in {".pdf", ".doc", ".docx", ".wps"}:
        return [url] if is_allowed_official_url(url) else []
    try:
        response = session.get(url, headers=OFFICIAL_REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        text = response.text or ""
    except Exception:
        try:
            text, _, _ = fetch_official_page_content(url, timeout)
        except Exception:
            return [url] if is_allowed_official_url(url) else []
    attachment_urls = extract_attachment_urls(text, url)
    return attachment_urls or ([url] if is_allowed_official_url(url) else [])


def write_text_official_page(target_path: Path, item: dict[str, Any], text: str, source_url: str) -> int:
    html = "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            f"<title>{escape(item.get('title') or source_url)}</title>",
            f"<h1>{escape(item.get('title') or source_url)}</h1>",
            f"<p>来源：<a href=\"{escape(source_url)}\">{escape(source_url)}</a></p>",
            f"<p>抓取时间：{escape(datetime.now().isoformat(timespec='seconds'))}</p>",
            "<pre>",
            escape(text),
            "</pre>",
        ]
    )
    data = html.encode("utf-8")
    target_path.write_bytes(data)
    return len(data)


def write_bounded_response(response: requests.Response, target_path: Path, max_bytes: int) -> int:
    total = 0
    with target_path.open("wb") as file:
        if hasattr(response, "iter_content"):
            chunks = response.iter_content(chunk_size=65536)
        else:
            chunks = [getattr(response, "content", b"")]
        for chunk in chunks:
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise LawChatHostError(f"官方候选文件超过最大下载限制：{max_bytes} bytes")
            file.write(chunk)
    return total


def download_response_or_reader_text(
    session: requests.Session,
    download_url: str,
    target_path: Path,
    item: dict[str, Any],
    timeout: int,
    max_bytes: int,
) -> tuple[int, str, str]:
    try:
        response = session.get(download_url, headers=OFFICIAL_REQUEST_HEADERS, timeout=timeout, stream=True)
        response.raise_for_status()
        written = write_bounded_response(response, target_path, max_bytes)
        content_type = (getattr(response, "headers", {}) or {}).get("Content-Type", "")
        return written, "official_direct", content_type
    except Exception as exc:
        if extension_from_url_or_content_type(download_url) != ".html":
            raise
        text, metadata, _ = fetch_official_page_content(download_url, timeout)
        content_text = clean_html_text(text) if "<html" in text[:500].lower() else clean_markdown_text(text)
        if not content_text:
            raise LawChatHostError(f"官方页面下载失败且阅读器未返回正文：{exc}") from exc
        written = write_text_official_page(target_path, item, content_text, download_url)
        if written > max_bytes:
            target_path.unlink(missing_ok=True)
            raise LawChatHostError(f"官方候选文件超过最大下载限制：{max_bytes} bytes")
        return written, metadata.get("fetched_via") or "reader_fallback", "text/html; charset=utf-8"


def download_official_update_candidates(
    official_items: list[dict[str, Any]],
    comparison: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not should_download_official_candidates(comparison, config):
        return []
    root = Path(config.get("official_updates_dir") or OFFICIAL_UPDATES_PENDING_DIR)
    for dirname in ("pdf", "word", "html", "metadata"):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.trust_env = True
    timeout = int(config.get("download_timeout") or 20)
    max_bytes = int(config.get("max_download_bytes") or 50_000_000)
    records = []

    for item in official_items[:3]:
        download_urls = discover_download_urls(item, session, timeout)
        if not download_urls:
            continue
        download_url = download_urls[0]
        try:
            ext = extension_from_url_or_content_type(download_url)
            dirname = subdir_for_extension(ext)
            url_hash = hashlib.sha256(download_url.encode("utf-8")).hexdigest()[:12]
            stem = safe_filename(f"{item.get('date') or 'date_unknown'}_{item.get('title') or item.get('source') or 'official'}")
            target_path = root / dirname / f"{stem}_{url_hash}{ext}"
            written, fetched_via, content_type = download_response_or_reader_text(
                session,
                download_url,
                target_path,
                item,
                timeout,
                max_bytes,
            )
            actual_ext = extension_from_url_or_content_type(download_url, content_type)
            if actual_ext != ext:
                actual_dirname = subdir_for_extension(actual_ext)
                actual_target_path = root / actual_dirname / f"{stem}_{url_hash}{actual_ext}"
                actual_target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.replace(actual_target_path)
                target_path = actual_target_path
                dirname = actual_dirname
            metadata = {
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "download_url": download_url,
                "file_path": str(target_path),
                "file_type": dirname,
                "fetched_via": fetched_via,
                "publish_date": item.get("date"),
                "comparison_status": comparison.get("status"),
                "comparison_reason": comparison.get("reason"),
                "local_matches": comparison.get("local_items") or [],
                "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                "bytes": written,
            }
            metadata_path = root / "metadata" / f"{target_path.stem}.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append({**metadata, "metadata_path": str(metadata_path)})
        except Exception as exc:
            records.append(
                {
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "download_url": download_url,
                    "error": str(exc),
                    "comparison_status": comparison.get("status"),
                }
            )
    return records


def prepare_official_update_candidates(
    local_result: dict[str, Any] | None,
    web_result: dict[str, Any] | None,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not web_result:
        return None, []
    comparison = compare_local_and_official(local_result, web_result)
    official_items = (web_result or {}).get("results") or []
    downloads = download_official_update_candidates(official_items, comparison, config)
    return comparison, downloads


def append_official_update_note(
    answer: str,
    comparison: dict[str, Any] | None,
    downloaded_updates: list[dict[str, Any]] | None,
) -> str:
    if not comparison or comparison.get("status") in {None, "no_official_result"}:
        return answer
    lines = ["", "本地/官网时效对比：", f"- 判断：{comparison.get('reason') or comparison.get('status')}"]
    local_items = comparison.get("local_items") or []
    official_items = comparison.get("official_items") or []
    if local_items:
        first_local = local_items[0]
        lines.append(
            f"- 本地库：{first_local.get('doc_title') or '未知文件'}，"
            f"日期 {first_local.get('date') or '未知'}，"
            f"状态 {first_local.get('status') or '未知'}，"
            f"chunk_id={first_local.get('chunk_id') or '未知'}"
        )
    else:
        lines.append("- 本地库：未检索到对应依据。")
    if official_items:
        first_official = official_items[0]
        lines.append(
            f"- 官网候选：{first_official.get('source') or '官方来源'}《{first_official.get('title') or '未命名'}》，"
            f"日期 {first_official.get('date') or '未知'}，{first_official.get('url') or ''}"
        )
    if downloaded_updates:
        saved = [record for record in downloaded_updates if record.get("file_path")]
        if saved:
            lines.append("- 待入库文件：")
            for record in saved[:3]:
                lines.append(f"  - {record.get('file_path')}（metadata: {record.get('metadata_path')}）")
        errors = [record for record in downloaded_updates if record.get("error")]
        for record in errors[:2]:
            lines.append(f"- 候选文件下载失败：{record.get('download_url') or record.get('url')}；{record.get('error')}")
    return (answer.rstrip() + "\n" + "\n".join(lines)).strip()


def should_use_official_web(user_message: str, local_result: dict[str, Any] | None, config: dict[str, Any]) -> bool:
    if not config.get("web_enabled", True):
        return False
    text = message_content_to_text(user_message)
    if any(keyword in text for keyword in OFFICIAL_WEB_TRIGGER_KEYWORDS):
        return True
    if not local_result:
        return True
    if local_result.get("ok") is False:
        return True
    return not bool(local_result.get("results"))


def tool_message_from_result(tool_entry: dict[str, Any], tool_call_id: str | None = None) -> dict[str, Any]:
    message = {
        "role": "tool",
        "name": tool_entry["name"],
        "content": json.dumps(tool_entry["result"], ensure_ascii=False),
    }
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def run_chat_turn(user_message: str, chat_history: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    law_related = is_law_related_question(user_message)
    if config.get("no_tools"):
        return run_fallback_evidence_answer(user_message, chat_history, config, initial_messages(user_message, chat_history))

    messages = initial_messages(user_message, chat_history)
    tools = build_tools_schema()
    tool_history: list[dict[str, Any]] = []
    used_chunks: dict[str, dict[str, Any]] = {}
    used_web_sources: dict[str, dict[str, Any]] = {}

    for _ in range(int(config.get("max_agent_steps") or 6)):
        response = ollama_chat(messages, tools, config, stream=False)
        assistant_message = response.get("message") or {"role": "assistant", "content": ""}
        tool_calls = extract_tool_calls(response)
        if tool_calls:
            messages.append(assistant_message)
            for call in tool_calls:
                tool_entry = execute_tool_call(call, config)
                tool_history.append(tool_entry)
                collect_used_chunks_from_tool_result(tool_entry, used_chunks)
                collect_used_web_sources_from_tool_result(tool_entry, used_web_sources)
                messages.append(tool_message_from_result(tool_entry, call.get("id")))
            continue

        local_result = latest_tool_result(tool_history, "search_laws")
        has_web_search = any(entry.get("name") == "official_web_search" for entry in tool_history)
        if law_related and local_result and not has_web_search and should_use_official_web(user_message, local_result, config):
            tool_entry = execute_tool_call(
                {"name": "official_web_search", "arguments": {"query": user_message, "limit": config.get("top_k", 5)}},
                config,
            )
            tool_history.append(tool_entry)
            collect_used_web_sources_from_tool_result(tool_entry, used_web_sources)
            messages.append(tool_message_from_result(tool_entry))
            continue

        answer = assistant_message.get("content") or ""
        if law_related and not tool_history:
            return run_fallback_evidence_answer(user_message, chat_history, config, messages)
        evidence_comparison, downloaded_official_updates = prepare_official_update_candidates(
            latest_tool_result(tool_history, "search_laws"),
            latest_tool_result(tool_history, "official_web_search"),
            config,
        )
        answer = append_official_update_note(answer, evidence_comparison, downloaded_official_updates)
        messages.append({"role": "assistant", "content": answer})
        citation_check = validate_answer_citations(
            answer,
            used_chunks,
            law_related=law_related,
            used_web_sources=used_web_sources,
        )
        result = {
            "answer": answer,
            "used_chunks": used_chunks,
            "used_web_sources": used_web_sources,
            "evidence_comparison": evidence_comparison,
            "downloaded_official_updates": downloaded_official_updates,
            "tool_history": tool_history,
            "citation_check": citation_check,
            "messages": messages,
        }
        if config.get("save_log"):
            save_log(user_message, result, config)
        return result

    messages.append({"role": "user", "content": "请停止调用工具，基于已经返回的工具证据给出最终答案。"})
    response = ollama_chat(messages, None, config, stream=False)
    answer = (response.get("message") or {}).get("content") or ""
    evidence_comparison, downloaded_official_updates = prepare_official_update_candidates(
        latest_tool_result(tool_history, "search_laws"),
        latest_tool_result(tool_history, "official_web_search"),
        config,
    )
    answer = append_official_update_note(answer, evidence_comparison, downloaded_official_updates)
    messages.append({"role": "assistant", "content": answer})
    citation_check = validate_answer_citations(
        answer,
        used_chunks,
        law_related=law_related,
        used_web_sources=used_web_sources,
    )
    result = {
        "answer": answer,
        "used_chunks": used_chunks,
        "used_web_sources": used_web_sources,
        "evidence_comparison": evidence_comparison,
        "downloaded_official_updates": downloaded_official_updates,
        "tool_history": tool_history,
        "citation_check": citation_check,
        "messages": messages,
    }
    if config.get("save_log"):
        save_log(user_message, result, config)
    return result


def run_fallback_evidence_answer(
    user_message: str,
    chat_history: list[dict[str, Any]],
    config: dict[str, Any],
    previous_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tool_entry = {
        "name": "search_laws",
        "arguments": {"query": user_message, "top_k": config["top_k"]},
        "result": search_laws(user_message, top_k=config["top_k"], config=config),
    }
    tool_history = [tool_entry]
    used_chunks: dict[str, dict[str, Any]] = {}
    used_web_sources: dict[str, dict[str, Any]] = {}
    collect_used_chunks_from_tool_result(tool_entry, used_chunks)
    web_result = None
    if should_use_official_web(user_message, tool_entry["result"], config):
        web_entry = {
            "name": "official_web_search",
            "arguments": {"query": user_message, "limit": min(int(config.get("top_k") or 5), 5)},
            "result": official_web_search(user_message, limit=min(int(config.get("top_k") or 5), 5), config=config),
        }
        tool_history.append(web_entry)
        collect_used_web_sources_from_tool_result(web_entry, used_web_sources)
        web_result = web_entry["result"]
    evidence_comparison, downloaded_official_updates = prepare_official_update_candidates(
        tool_entry["result"],
        web_result,
        config,
    )
    evidence_payload = {
        "local_search": tool_entry["result"],
        "official_web_search": web_result,
        "evidence_comparison": evidence_comparison,
        "downloaded_official_updates": downloaded_official_updates,
    }
    evidence = json.dumps(evidence_payload, ensure_ascii=False, indent=2)
    messages = list(previous_messages or initial_messages(user_message, chat_history))
    messages.append(
        {
            "role": "user",
            "content": "模型未稳定调用工具。以下是 Host 主动检索本地法规数据库与官方白名单网站得到的 evidence。请先自然地直接回答，再列依据；本地依据引用 chunk_id，官网依据引用来源和 URL；若证据不足，回答“未检索到明确依据”。\n"
            + evidence,
        }
    )
    generation_warning = None
    try:
        response = ollama_chat(messages, None, config, stream=False)
        answer = (response.get("message") or {}).get("content") or ""
    except Exception as exc:
        answer = ""
        generation_warning = f"LLM 生成失败，已使用检索证据合成保守答案：{exc}"
    if not answer.strip():
        answer = synthesize_answer_from_evidence(
            tool_entry["result"],
            web_result,
            evidence_comparison,
            downloaded_official_updates,
        )
    else:
        answer = append_official_update_note(answer, evidence_comparison, downloaded_official_updates)
    messages.append({"role": "assistant", "content": answer})
    citation_check = validate_answer_citations(answer, used_chunks, law_related=True, used_web_sources=used_web_sources)
    if generation_warning:
        citation_check.setdefault("warnings", []).append(generation_warning)
    result = {
        "answer": answer,
        "used_chunks": used_chunks,
        "used_web_sources": used_web_sources,
        "evidence_comparison": evidence_comparison,
        "downloaded_official_updates": downloaded_official_updates,
        "tool_history": tool_history,
        "citation_check": citation_check,
        "messages": messages,
    }
    if config.get("save_log"):
        save_log(user_message, result, config)
    return result


def synthesize_answer_from_evidence(
    search_result: dict[str, Any],
    web_result: dict[str, Any] | None = None,
    evidence_comparison: dict[str, Any] | None = None,
    downloaded_official_updates: list[dict[str, Any]] | None = None,
) -> str:
    results = search_result.get("results") if isinstance(search_result, dict) else None
    web_results = web_result.get("results") if isinstance(web_result, dict) else None
    if not results and not web_results:
        return "未检索到明确依据。请补充更具体的法规名称、业务场景或发文机关后再查。"

    lines = ["我检索到了可用依据，关键结论请以原文为准。", "", "依据："]
    if results:
        first = results[0]
        page = format_page_range(first)
        title = first.get("doc_title") or "未知文件"
        article = first.get("article_no") or "未知条号"
        chunk_id = first.get("chunk_id") or ""
        preview = (first.get("content_preview") or "").strip()
        if len(preview) > 220:
            preview = preview[:220].rstrip() + "……"
        lines.append(f"1. 本地库：《{title}》{article}，第{page or '未知'}页，chunk_id={chunk_id}：{preview}")
    if web_results:
        index = 2 if results else 1
        first_web = web_results[0]
        source = first_web.get("source") or "官方来源"
        title = first_web.get("title") or "未命名文件"
        date = first_web.get("date") or "日期未知"
        url = first_web.get("url") or ""
        snippet = (first_web.get("snippet") or "").strip()
        if len(snippet) > 180:
            snippet = snippet[:180].rstrip() + "……"
        lines.append(f"{index}. 官网联网：{source}《{title}》，{date}，{url}：{snippet}")
    lines.append("")
    lines.append("注意：官网联网结果是实时检索线索；作正式依据前，建议打开 URL 核对全文和附件。")
    return append_official_update_note("\n".join(lines), evidence_comparison, downloaded_official_updates)


def extract_cited_chunk_ids(answer: str) -> list[str]:
    found = re.findall(r"chunk_id\s*[=:：]\s*([A-Za-z0-9_.:-]+)", answer)
    seen = set()
    result = []
    for chunk_id in found:
        cleaned = chunk_id.strip().rstrip("，。；;,.、")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def extract_cited_urls(answer: str) -> list[str]:
    found = re.findall(r"https?://[^\s，。；；,、：:)）\]]+", answer or "")
    seen = set()
    result = []
    for url in found:
        cleaned = url.rstrip("。；;，,、")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def validate_answer_citations(
    answer: str,
    used_chunks: dict[str, Any],
    law_related: bool = True,
    used_web_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cited = extract_cited_chunk_ids(answer)
    cited_urls = extract_cited_urls(answer)
    used_web_sources = used_web_sources or {}
    warnings = []
    if law_related and not cited and not cited_urls and "未检索到明确依据" not in answer:
        warnings.append("法规相关回答未引用任何 chunk_id 或官方 URL")
    unknown = [chunk_id for chunk_id in cited if chunk_id not in used_chunks]
    for chunk_id in unknown:
        warnings.append(f"回答引用了未检索过的 chunk_id：{chunk_id}")
    unknown_urls = [url for url in cited_urls if url not in used_web_sources]
    for url in unknown_urls:
        warnings.append(f"回答引用了未检索过的官方 URL：{url}")
    return {"cited_chunk_ids": cited, "cited_urls": cited_urls, "warnings": warnings}


def format_page_range(chunk: dict[str, Any]) -> str:
    start = chunk.get("page_start")
    end = chunk.get("page_end")
    if start and end and start != end:
        return f"{start}-{end}"
    if start:
        return str(start)
    if end:
        return str(end)
    return ""


def format_used_chunks(used_chunks: dict[str, dict[str, Any]]) -> str:
    if not used_chunks:
        return "使用证据：无"
    lines = ["使用证据："]
    for chunk_id, chunk in used_chunks.items():
        lines.append(
            f"- chunk_id={chunk_id} | {chunk.get('doc_title') or ''} | {chunk.get('article_no') or ''} | 页码 {format_page_range(chunk)}"
        )
    return "\n".join(lines)


def format_used_web_sources(used_web_sources: dict[str, dict[str, Any]]) -> str:
    if not used_web_sources:
        return "官网联网证据：无"
    lines = ["官网联网证据："]
    for url, source in used_web_sources.items():
        lines.append(
            f"- {source.get('source') or '官方来源'} | {source.get('title') or ''} | {source.get('date') or '日期未知'} | {url}"
        )
    return "\n".join(lines)


def run_interactive(config: dict[str, Any]) -> None:
    print("法规 Agent Host 已启动")
    print(f"LLM：{config['llm_model']}")
    print(f"Embedding：{config['embedding_model']}")
    print(f"SQLite：{config['sqlite_db']}")
    print(f"Qdrant：{config['collection']}")
    print("请输入问题，输入 exit 退出。")
    chat_history: list[dict[str, Any]] = []
    while True:
        try:
            user_message = input("你：").strip()
        except EOFError:
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit", "q"}:
            break
        result = run_chat_turn(user_message, chat_history, config)
        print("助手：")
        print(result["answer"])
        print(format_used_chunks(result.get("used_chunks") or {}))
        print(format_used_web_sources(result.get("used_web_sources") or {}))
        warnings = result.get("citation_check", {}).get("warnings") or []
        if warnings:
            print("引用检查 warning：")
            for warning in warnings:
                print(f"- {warning}")
        if config.get("debug"):
            print("工具调用过程：")
            print(json.dumps(result.get("tool_history") or [], ensure_ascii=False, indent=2))
        chat_history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result["answer"]},
            ]
        )
        chat_history = trim_chat_history(chat_history)


def build_openai_completion(answer: str, model_name: str, created: int) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-local-law-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def iter_openai_sse(answer: str, model_name: str, created: int):
    response_id = f"chatcmpl-local-law-{created}"
    chunks = [
        {"delta": {"role": "assistant"}, "finish_reason": None},
        {"delta": {"content": answer}, "finish_reason": None},
        {"delta": {}, "finish_reason": "stop"},
    ]
    for chunk in chunks:
        payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": chunk["delta"],
                    "finish_reason": chunk["finish_reason"],
                }
            ],
        }
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    yield "data: [DONE]\n\n"


def create_fastapi_app(config: dict[str, Any]):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise LawChatHostError(DEPENDENCY_MESSAGE) from exc

    app = FastAPI(title="Local Law Agent Host")

    class ChatRequest(BaseModel):
        message: str
        history: list[dict[str, Any]] = []

    class OpenAIChatRequest(BaseModel):
        model: str | None = None
        messages: list[dict[str, Any]]
        temperature: float | None = 0.1
        stream: bool | None = False

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "llm_model": config["llm_model"],
            "sqlite_db_exists": Path(config["sqlite_db"]).exists(),
            "qdrant_collection": config["collection"],
        }

    @app.get("/tools")
    def tools():
        return {"tools": build_tools_schema()}

    @app.post("/chat")
    def chat(request: ChatRequest = Body(...)):
        try:
            return run_chat_turn(
                message_content_to_text(request.message),
                normalize_host_history(request.history),
                config,
            )
        except LawChatHostError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/chat/completions")
    def openai_chat(request: OpenAIChatRequest = Body(...)):
        messages = request.messages or []
        latest, history = extract_latest_user_message(messages)
        if not latest:
            answer = "未收到用户问题"
            result = {"answer": answer, "used_chunks": {}, "tool_history": [], "citation_check": {}, "messages": []}
        else:
            try:
                result = run_chat_turn(latest, history, config)
            except LawChatHostError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            answer = result["answer"]
        created = int(time.time())
        model_name = request.model or "law-agent-qwen3.6"
        if request.stream:
            return StreamingResponse(
                iter_openai_sse(answer, model_name, created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return build_openai_completion(answer, model_name, created)

    return app


def run_server(config: dict[str, Any]) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise LawChatHostError(DEPENDENCY_MESSAGE) from exc
    app = create_fastapi_app(config)
    uvicorn.run(app, host=config["host"], port=int(config["port"]))


def save_log(user_message: str, result: dict[str, Any], config: dict[str, Any]) -> Path:
    AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGENT_LOG_DIR / f"law_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "user_message": user_message,
        "answer": result.get("answer"),
        "used_chunks": result.get("used_chunks"),
        "used_web_sources": result.get("used_web_sources"),
        "evidence_comparison": result.get("evidence_comparison"),
        "downloaded_official_updates": result.get("downloaded_official_updates"),
        "tool_history": result.get("tool_history"),
        "citation_check": result.get("citation_check"),
        "messages": result.get("messages"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    try:
        args = parse_args()
        config = build_config(args)
        if config["top_k"] < 1 or config["keyword_top_k"] < 1 or config["semantic_top_k"] < 1:
            raise LawChatHostError("--top-k、--keyword-top-k、--semantic-top-k 必须大于 0")
        if config["max_agent_steps"] < 1:
            raise LawChatHostError("--max-agent-steps 必须大于 0")
        preflight_check(config)
        if config.get("serve"):
            run_server(config)
        else:
            run_interactive(config)
    except LawChatHostError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
