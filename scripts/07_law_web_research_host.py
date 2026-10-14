#!/usr/bin/env python3
"""
Web-Augmented 法规 Agent Host。

本脚本在 06_law_chat_host.py 的本地法规 RAG 能力上增加 SearXNG/Tavily 联网检索、
网页正文读取、问题路由和联网引用检查。它不重建索引，也不修改本地数据库。
"""

import argparse
import importlib.util
import ipaddress
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - surfaced by preflight_check.
    BeautifulSoup = None

try:
    import trafilatura
except ImportError:  # pragma: no cover - web_fetch falls back to BeautifulSoup.
    trafilatura = None

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover - surfaced by preflight_check.
    QdrantClient = None


ROOT_DIR = Path(os.environ.get("CDC_LAW_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUTPUT_DIR = ROOT_DIR / "_rag_output"
SQLITE_DB = OUTPUT_DIR / "law_rag.db"
AGENT_LOG_DIR = OUTPUT_DIR / "agent_logs"
BASE_HOST_SCRIPT = ROOT_DIR / "scripts" / "06_law_chat_host.py"
HYBRID_SEARCH_SCRIPT = ROOT_DIR / "scripts" / "04_hybrid_search.py"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.6:27b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "law_chunks_local"
DEFAULT_SEARXNG_URL = "http://127.0.0.1:8080"
DEFAULT_WEB_TOP_K = 5

DEPENDENCY_MESSAGE = (
    "请先安装依赖：\n"
    "pip install requests beautifulsoup4 trafilatura fastapi uvicorn pydantic qdrant-client tqdm"
)

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

WEB_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
}

LAW_KEYWORDS = [
    "法规",
    "法律",
    "条例",
    "办法",
    "规定",
    "制度",
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
    "能否",
    "是否",
]

WEB_KEYWORDS = [
    "最新",
    "现在",
    "目前",
    "近期",
    "官方",
    "政策解读",
    "通知",
    "公告",
    "公开资料",
    "网上",
    "新闻",
    "进展",
    "有没有新规定",
    "是否修订",
    "现行版本",
    "外部资料",
    "国家规定",
    "卫健委",
    "国务院",
    "市场监管",
    "人社部",
    "疾控局",
]

OFFICIAL_KEYWORDS = [
    "官方",
    "国家",
    "国务院",
    "卫健委",
    "疾控",
    "市场监管",
    "人社部",
    "人大",
    "法院",
    "政策",
    "通知",
    "公告",
    "条例",
    "法律",
    "法规",
]

OFFICIAL_SITE_FILTERS = [
    "gov.cn",
    "nhc.gov.cn",
    "chinacdc.cn",
    "samr.gov.cn",
    "mohrss.gov.cn",
    "npc.gov.cn",
    "court.gov.cn",
]

BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

_base_host_module = None
hybrid_search = None


def load_base_host_module():
    global _base_host_module
    if _base_host_module is None:
        if not BASE_HOST_SCRIPT.exists():
            raise RuntimeError(f"找不到基础脚本：{BASE_HOST_SCRIPT}")
        spec = importlib.util.spec_from_file_location("law_chat_host_base", BASE_HOST_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _base_host_module = module
    return _base_host_module


_base_host = load_base_host_module()
LawChatHostError = _base_host.LawChatHostError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web-Augmented 本地法规 Agent Host")
    parser.add_argument("--interactive", "-interactive", action="store_true", help="命令行聊天模式")
    parser.add_argument("--serve", action="store_true", help="HTTP 服务模式")
    parser.add_argument("--host", "-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--web-provider", choices=["searxng", "tavily", "none"], default="searxng")
    parser.add_argument("--searxng-url", default=DEFAULT_SEARXNG_URL)
    parser.add_argument("--web-top-k", type=int, default=DEFAULT_WEB_TOP_K)
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
        "web_provider": args.web_provider,
        "searxng_url": args.searxng_url,
        "web_top_k": int(args.web_top_k),
        "web_enabled": args.web_provider != "none",
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
        "tavily_api_key": os.environ.get("TAVILY_API_KEY"),
    }


def check_dependencies(require_server: bool = False) -> None:
    required = ["requests", "bs4", "qdrant_client", "tqdm"]
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
    if trafilatura is None:
        print("提示：trafilatura 未安装，web_fetch 将使用 BeautifulSoup fallback。建议执行：pip install trafilatura", file=sys.stderr)


def load_hybrid_search_module():
    global hybrid_search
    module = _base_host.load_hybrid_search_module()
    hybrid_search = _base_host.hybrid_search
    return module


def preflight_check(config: dict[str, Any]) -> None:
    check_dependencies(require_server=bool(config.get("serve")))
    db_path = Path(config["sqlite_db"])
    if not db_path.exists():
        raise LawChatHostError("找不到 SQLite 数据库，请先运行：\npython scripts/03_build_indexes.py")
    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')").fetchall()
        }
    finally:
        conn.close()
    if "law_chunks" not in tables:
        raise LawChatHostError("SQLite 数据库中找不到 law_chunks 表，请先运行：\npython scripts/03_build_indexes.py")
    if not HYBRID_SEARCH_SCRIPT.exists():
        raise LawChatHostError(f"找不到混合检索脚本：{HYBRID_SEARCH_SCRIPT}")
    load_hybrid_search_module()

    tags = _base_host.get_ollama_model_names(config["ollama_url"])
    if config["llm_model"] not in tags:
        raise LawChatHostError(f"模型 {config['llm_model']} 不存在，请先执行：\nollama pull {config['llm_model']}")
    if config["embedding_model"] not in tags:
        raise LawChatHostError(f"模型 {config['embedding_model']} 不存在，请先执行：\nollama pull {config['embedding_model']}")

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

    provider = config.get("web_provider")
    if provider == "searxng":
        probe = web_search("卫生健康", top_k=1, searxng_url=config["searxng_url"])
        if not probe.get("ok"):
            config["web_available"] = False
            print("SearXNG 不可用，联网搜索功能将不可用；本地法规库仍可使用。", file=sys.stderr)
    elif provider == "tavily" and not (config.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY")):
        raise LawChatHostError("Tavily 搜索需要设置环境变量 TAVILY_API_KEY。")


def get_system_prompt() -> str:
    return """你是一个本地法律法规研究型 Agent。
你有两类资料来源：
一、本地法规库工具：
- search_laws
- read_chunk
- check_validity
- list_categories
二、联网资料工具：
- web_search
- official_web_search
- web_fetch
你必须根据用户问题决定使用哪些工具。
规则：
1. 涉及本单位已有法规、制度、条例、办法、职责、处罚、报告义务、审批、备案、许可、监督检查等问题，必须先查本地法规库。
2. 涉及“最新、目前、官方、公开资料、政策解读、是否修订、网上资料”等问题，必须进行联网搜索。
3. 涉及法律法规、政策、卫生监督、疾控、医疗机构、传染病、职业病、食品安全等问题时，优先使用 official_web_search。
4. 联网结果不能替代本地法规库依据。请区分“本地法规库依据”和“联网公开资料”。
5. 关键结论必须给出处。
6. 本地法规库引用必须包含：文件名、条号、页码、chunk_id。
7. 联网资料引用必须包含：网页标题、URL、来源站点。
8. 如果证据不足，必须明确说“未检索到明确依据”。
9. 不要凭模型常识编造法规依据。
10. 如果本地法规库与联网公开资料不一致，请明确提示可能存在版本差异，并建议人工核验正式文本。
最终回答格式：
结论：
……
本地法规库依据：
1. 《文件名》第X条，第Y页，chunk_id=...：……
联网公开资料：
1. 标题：……
来源：……
URL：……
摘要：……
综合判断：
……
注意事项：
……
未覆盖或证据不足：
……"""


def build_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_laws",
                "description": "检索本地法律法规、条例、制度、办法、规范性文件数据库。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "status": {"type": "string"},
                        "category": {"type": "string"},
                        "doc_title": {"type": "string"},
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
                    "properties": {"chunk_id": {"type": "string"}},
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
                    "properties": {"limit": {"type": "integer"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "通过 SearXNG 检索联网公开资料，可带 site_filter。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "site_filter": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "official_web_search",
                "description": "优先检索 gov.cn、卫健委、中国疾控、市场监管、人社部、人大、法院等官方站点。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "读取网页正文，拒绝 file、localhost、127.0.0.1 和内网地址。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_chars": {"type": "integer"},
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def ollama_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, config: dict[str, Any], stream: bool = False) -> dict[str, Any]:
    return _base_host.ollama_chat(messages, tools, config, stream=stream)


def classify_question_need(question: str) -> dict[str, Any]:
    text = question or ""
    need_law_db = any(word in text for word in LAW_KEYWORDS)
    need_web = any(word in text for word in WEB_KEYWORDS) or bool(re.search(r"https?://", text))
    need_official_web = need_web and (need_law_db or any(word in text for word in OFFICIAL_KEYWORDS))
    reasons = []
    if need_law_db:
        reasons.append("问题涉及法规、卫生监督或报告义务，需要查本地法规库")
    if need_web:
        reasons.append("问题涉及最新、官方或网上公开资料，需要联网搜索")
    if need_official_web:
        reasons.append("问题涉及官方政策或法规资料，优先检索官方站点")
    if not reasons:
        reasons.append("未识别出必须检索法规库或联网资料的关键词，可普通回答")
    return {
        "need_law_db": need_law_db,
        "need_web": need_web,
        "need_official_web": need_official_web,
        "reason": "；".join(reasons),
    }


def is_law_related_question(question: str) -> bool:
    return bool(classify_question_need(question).get("need_law_db"))


def need_web_search(question: str) -> bool:
    return bool(classify_question_need(question).get("need_web"))


def search_laws(
    query: str,
    top_k: int = 8,
    status: str | None = None,
    category: str | None = None,
    doc_title: str | None = None,
    access_level: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _base_host.search_laws(
        query=query,
        top_k=top_k,
        status=status,
        category=category,
        doc_title=doc_title,
        access_level=access_level,
        config=config,
    )


def read_chunk(chunk_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return _base_host.read_chunk(chunk_id=chunk_id, config=config)


def check_validity(
    doc_id: str | None = None,
    doc_title: str | None = None,
    chunk_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _base_host.check_validity(doc_id=doc_id, doc_title=doc_title, chunk_id=chunk_id, config=config)


def list_categories(limit: int = 100, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return _base_host.list_categories(limit=limit, config=config)


def normalize_search_result(item: dict[str, Any]) -> dict[str, Any]:
    title = item.get("title") or item.get("name") or ""
    url = item.get("url") or item.get("link") or ""
    snippet = item.get("snippet") or item.get("content") or item.get("description") or ""
    return {
        "title": str(title),
        "url": str(url),
        "snippet": str(snippet),
        "source": item.get("source") or item.get("engine") or item.get("provider") or "web",
    }


def web_search(
    query: str,
    top_k: int = DEFAULT_WEB_TOP_K,
    searxng_url: str = DEFAULT_SEARXNG_URL,
    site_filter: str | None = None,
) -> dict[str, Any]:
    top_k = max(1, int(top_k or DEFAULT_WEB_TOP_K))
    search_query = f"site:{site_filter} {query}".strip() if site_filter else query
    try:
        session = requests.Session()
        session.trust_env = True
        response = session.get(
            searxng_url.rstrip("/") + "/search",
            params={"q": search_query, "format": "json", "language": "zh-CN"},
            headers=WEB_REQUEST_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        rows = data.get("results") if isinstance(data, dict) else []
        results = [normalize_search_result(item) for item in (rows or [])[:top_k] if isinstance(item, dict)]
        return {
            "tool": "web_search",
            "provider": "searxng",
            "ok": True,
            "query": query,
            "site_filter": site_filter,
            "result_count": len(results),
            "results": results,
            "error": None,
        }
    except Exception as exc:
        return {
            "tool": "web_search",
            "provider": "searxng",
            "ok": False,
            "query": query,
            "site_filter": site_filter,
            "result_count": 0,
            "results": [],
            "error": f"未连接到 SearXNG。请先启动 SearXNG，或改用 --web-provider tavily。详情：{exc}",
        }


def tavily_search(query: str, top_k: int = DEFAULT_WEB_TOP_K, api_key: str | None = None) -> dict[str, Any]:
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        return {
            "tool": "tavily_search",
            "provider": "tavily",
            "ok": False,
            "query": query,
            "result_count": 0,
            "results": [],
            "error": "未配置 TAVILY_API_KEY，无法使用 Tavily。请设置环境变量 TAVILY_API_KEY。",
        }
    try:
        session = requests.Session()
        session.trust_env = True
        response = session.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max(1, int(top_k or DEFAULT_WEB_TOP_K)),
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in (data.get("results") or [])[: max(1, int(top_k or DEFAULT_WEB_TOP_K))]:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("snippet") or ""
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "snippet": content,
                    "content": content,
                    "score": item.get("score"),
                }
            )
        return {
            "tool": "tavily_search",
            "provider": "tavily",
            "ok": True,
            "query": query,
            "result_count": len(results),
            "results": results,
            "error": None,
        }
    except Exception as exc:
        return {
            "tool": "tavily_search",
            "provider": "tavily",
            "ok": False,
            "query": query,
            "result_count": 0,
            "results": [],
            "error": f"Tavily 检索失败：{exc}",
        }


def provider_web_search(query: str, top_k: int, config: dict[str, Any], site_filter: str | None = None) -> dict[str, Any]:
    provider = config.get("web_provider", "searxng")
    if provider == "none":
        return {"tool": "web_search", "ok": False, "query": query, "result_count": 0, "results": [], "error": "联网搜索已禁用"}
    if provider == "tavily":
        tavily_query = f"site:{site_filter} {query}".strip() if site_filter else query
        return tavily_search(tavily_query, top_k=top_k, api_key=config.get("tavily_api_key"))
    return web_search(query, top_k=top_k, searxng_url=config.get("searxng_url", DEFAULT_SEARXNG_URL), site_filter=site_filter)


def official_web_search(query: str, top_k: int = DEFAULT_WEB_TOP_K, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    top_k = max(1, int(top_k or DEFAULT_WEB_TOP_K))
    results = []
    warnings = []
    seen = set()
    for site in OFFICIAL_SITE_FILTERS:
        if len(results) >= top_k:
            break
        response = provider_web_search(query, top_k=top_k, config=config, site_filter=site)
        if not response.get("ok"):
            warnings.append(response.get("error") or f"{site} 检索失败")
            continue
        for item in response.get("results") or []:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": url,
                    "snippet": item.get("snippet") or item.get("content") or "",
                    "source": item.get("source") or response.get("provider") or "official_web",
                    "source_type": "official",
                }
            )
            if len(results) >= top_k:
                break
    return {
        "tool": "official_web_search",
        "ok": bool(results),
        "query": query,
        "result_count": len(results),
        "results": results,
        "warnings": warnings,
    }


def is_blocked_fetch_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return True, "只允许读取 http:// 或 https:// URL"
    host = (parsed.hostname or "").lower()
    if not host:
        return True, "URL 缺少主机名"
    if host in BLOCKED_HOSTS or host.endswith(".local"):
        return True, f"禁止访问本机或内网地址：{host}"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_unspecified or ip.is_link_local:
            return True, f"禁止访问本机或内网地址：{host}"
    except ValueError:
        pass
    return False, ""


def extract_text_with_bs4(html: str) -> tuple[str, str]:
    if BeautifulSoup is None:
        return "", ""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def web_fetch(url: str, max_chars: int = 5000) -> dict[str, Any]:
    blocked, reason = is_blocked_fetch_url(url)
    if blocked:
        return {"tool": "web_fetch", "url": url, "title": "", "text": "", "text_length": 0, "ok": False, "error": reason}
    try:
        session = requests.Session()
        session.trust_env = True
        response = session.get(url, headers=WEB_REQUEST_HEADERS, timeout=20)
        response.raise_for_status()
        html = response.text or ""
        title, fallback_text = extract_text_with_bs4(html)
        extracted = ""
        if trafilatura is not None:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        text = re.sub(r"\s+", " ", (extracted or fallback_text)).strip()
        text = text[: max(1, int(max_chars or 5000))]
        return {
            "tool": "web_fetch",
            "url": url,
            "title": title,
            "text": text,
            "text_length": len(text),
            "ok": True,
            "error": None,
        }
    except Exception as exc:
        return {"tool": "web_fetch", "url": url, "title": "", "text": "", "text_length": 0, "ok": False, "error": str(exc)}


def execute_tool_call(tool_call: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    name = tool_call.get("name")
    args = dict(tool_call.get("arguments") or {})
    try:
        if name == "search_laws":
            result = search_laws(
                query=args.get("query", ""),
                top_k=int(args.get("top_k") or config["top_k"]),
                status=args.get("status") or config.get("status"),
                category=args.get("category") or config.get("category"),
                doc_title=args.get("doc_title") or config.get("doc_title"),
                access_level=args.get("access_level") or config.get("access_level"),
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
        elif name == "web_search":
            result = provider_web_search(
                args.get("query", ""),
                top_k=int(args.get("top_k") or config.get("web_top_k") or DEFAULT_WEB_TOP_K),
                config=config,
                site_filter=args.get("site_filter"),
            )
        elif name == "official_web_search":
            result = official_web_search(
                args.get("query", ""),
                top_k=int(args.get("top_k") or args.get("limit") or config.get("web_top_k") or DEFAULT_WEB_TOP_K),
                config=config,
            )
        elif name == "web_fetch":
            result = web_fetch(args.get("url", ""), max_chars=int(args.get("max_chars") or 5000))
        else:
            result = {"ok": False, "error": f"未知工具：{name}"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    return {"name": name, "arguments": args, "result": result}


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return _base_host.extract_tool_calls(response)


def normalize_host_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return _base_host.normalize_host_history(messages)


def message_content_to_text(content: Any) -> str:
    return _base_host.message_content_to_text(content)


def extract_latest_user_message(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    return _base_host.extract_latest_user_message(messages)


def trim_chat_history(chat_history: list[dict[str, Any]], max_turn_messages: int = 20) -> list[dict[str, Any]]:
    return _base_host.trim_chat_history(chat_history, max_turn_messages=max_turn_messages)


def initial_messages(user_message: str, chat_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": get_system_prompt()}, *trim_chat_history(chat_history), {"role": "user", "content": user_message}]


def tool_message_from_result(tool_entry: dict[str, Any], tool_call_id: str | None = None) -> dict[str, Any]:
    return _base_host.tool_message_from_result(tool_entry, tool_call_id)


def collect_used_chunks_from_tool_result(tool_result: dict[str, Any], used_chunks: dict[str, dict[str, Any]]) -> None:
    _base_host.collect_used_chunks_from_tool_result(tool_result, used_chunks)


def collect_web_sources_from_tool_result(tool_result: dict[str, Any], web_sources: list[dict[str, Any]]) -> None:
    result = tool_result.get("result") if "result" in tool_result else tool_result
    if not isinstance(result, dict):
        return
    source_items = []
    if result.get("tool") == "web_fetch" and result.get("url"):
        source_items = [result]
    elif isinstance(result.get("results"), list):
        source_items = result["results"]
    seen = {item.get("url") for item in web_sources}
    for item in source_items:
        if not isinstance(item, dict) or not item.get("url") or item.get("url") in seen:
            continue
        seen.add(item["url"])
        web_sources.append(item)


def latest_tool_result(tool_history: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return _base_host.latest_tool_result(tool_history, name)


def extract_cited_chunk_ids(answer: str) -> list[str]:
    return _base_host.extract_cited_chunk_ids(answer)


def extract_cited_urls(answer: str) -> list[str]:
    return _base_host.extract_cited_urls(answer)


def validate_answer_citations(
    answer: str,
    used_chunks: dict[str, Any],
    law_related: bool = True,
    used_web_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _base_host.validate_answer_citations(answer, used_chunks, law_related=law_related, used_web_sources=used_web_sources)


def validate_web_citations(answer: str, web_sources: list[dict[str, Any]]) -> dict[str, Any]:
    warnings = []
    if "联网公开资料" in (answer or "") and not extract_cited_urls(answer):
        warnings.append("答案包含联网公开资料部分，但没有给出 URL")
    known_urls = {source.get("url") for source in web_sources if isinstance(source, dict)}
    for url in extract_cited_urls(answer):
        if url not in known_urls:
            warnings.append(f"回答引用了未检索到的 URL：{url}")
    return {"warnings": warnings}


def combined_citation_check(answer: str, used_chunks: dict[str, Any], web_sources: list[dict[str, Any]], classification: dict[str, Any]) -> dict[str, Any]:
    web_source_map = {item["url"]: item for item in web_sources if item.get("url")}
    local = validate_answer_citations(
        answer,
        used_chunks,
        law_related=bool(classification.get("need_law_db")),
        used_web_sources=web_source_map,
    )
    web = validate_web_citations(answer, web_sources)
    warnings = list(dict.fromkeys([*(local.get("warnings") or []), *(web.get("warnings") or [])]))
    return {**local, "web_warnings": web.get("warnings") or [], "warnings": warnings}


def used_chunks_list(used_chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**value, "chunk_id": key if not value.get("chunk_id") else value.get("chunk_id")} for key, value in used_chunks.items()]


def finalize_result(
    question: str,
    classification: dict[str, Any],
    answer: str,
    used_chunks: dict[str, dict[str, Any]],
    web_sources: list[dict[str, Any]],
    tool_history: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    citation_check = combined_citation_check(answer, used_chunks, web_sources, classification)
    result = {
        "question": question,
        "classification": classification,
        "answer": answer,
        "used_chunk_count": len(used_chunks),
        "used_chunks": used_chunks_list(used_chunks),
        "web_source_count": len(web_sources),
        "web_sources": web_sources,
        "tool_history": tool_history,
        "citation_check": citation_check,
        "messages": messages,
    }
    if config.get("save_log"):
        save_log(question, result, config)
    return result


def run_needed_fallback_tools(
    user_message: str,
    classification: dict[str, Any],
    config: dict[str, Any],
    tool_history: list[dict[str, Any]] | None = None,
    used_chunks: dict[str, dict[str, Any]] | None = None,
    web_sources: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    tool_history = list(tool_history or [])
    used_chunks = dict(used_chunks or {})
    web_sources = list(web_sources or [])
    has_law = any(entry.get("name") == "search_laws" for entry in tool_history)
    has_web = any(entry.get("name") in {"web_search", "official_web_search"} for entry in tool_history)
    if classification.get("need_law_db") and not has_law:
        entry = execute_tool_call({"name": "search_laws", "arguments": {"query": user_message, "top_k": config.get("top_k", 8)}}, config)
        tool_history.append(entry)
        collect_used_chunks_from_tool_result(entry, used_chunks)
    if classification.get("need_web") and not has_web:
        tool_name = "official_web_search" if classification.get("need_official_web") else "web_search"
        entry = execute_tool_call(
            {"name": tool_name, "arguments": {"query": user_message, "top_k": config.get("web_top_k", DEFAULT_WEB_TOP_K)}},
            config,
        )
        tool_history.append(entry)
        collect_web_sources_from_tool_result(entry, web_sources)
    return tool_history, used_chunks, web_sources


def synthesize_answer_from_evidence(user_message: str, used_chunks: dict[str, Any], web_sources: list[dict[str, Any]]) -> str:
    lines = ["结论：", "未检索到明确依据。" if not used_chunks and not web_sources else "请以以下检索证据为基础进行人工核验。"]
    lines.append("本地法规库依据：")
    if used_chunks:
        for chunk_id, chunk in list(used_chunks.items())[:3]:
            lines.append(
                f"1. 《{chunk.get('doc_title') or '未知文件'}》{chunk.get('article_no') or ''}，第{chunk.get('page_start') or '未知'}页，chunk_id={chunk_id}：{chunk.get('content_preview') or chunk.get('content') or ''}"
            )
    else:
        lines.append("未检索到明确依据。")
    lines.append("联网公开资料：")
    if web_sources:
        for source in web_sources[:3]:
            lines.append(
                f"1. 标题：{source.get('title') or '未命名'}\n来源：{source.get('source') or source.get('source_type') or 'web'}\nURL：{source.get('url')}\n摘要：{source.get('snippet') or source.get('text') or ''}"
            )
    else:
        lines.append("未检索到明确依据。")
    lines.extend(["综合判断：本地法规库与联网公开资料应区分使用，如存在版本差异，建议人工核验正式文本。", "注意事项：不要把联网公开资料直接替代本地正式法规依据。", "未覆盖或证据不足：以上为自动检索结果，关键结论仍需核验。"])
    return "\n".join(lines)


def run_fallback_evidence_answer(
    user_message: str,
    chat_history: list[dict[str, Any]],
    config: dict[str, Any],
    previous_messages: list[dict[str, Any]] | None = None,
    classification: dict[str, Any] | None = None,
    tool_history: list[dict[str, Any]] | None = None,
    used_chunks: dict[str, dict[str, Any]] | None = None,
    web_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    classification = classification or classify_question_need(user_message)
    tool_history, used_chunks, web_sources = run_needed_fallback_tools(
        user_message,
        classification,
        config,
        tool_history=tool_history,
        used_chunks=used_chunks,
        web_sources=web_sources,
    )
    evidence_payload = {
        "classification": classification,
        "tool_history": tool_history,
        "used_chunks": used_chunks_list(used_chunks),
        "web_sources": web_sources,
    }
    messages = list(previous_messages or initial_messages(user_message, chat_history))
    messages.append(
        {
            "role": "user",
            "content": "模型未稳定调用必要工具。以下是 Host 主动检索得到的 evidence。请严格按最终回答格式回答，区分本地法规库依据和联网公开资料，并引用 chunk_id 和 URL。\n"
            + json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        }
    )
    try:
        response = ollama_chat(messages, None, config, stream=False)
        answer = (response.get("message") or {}).get("content") or ""
    except Exception as exc:
        answer = ""
        messages.append({"role": "tool", "name": "llm_error", "content": str(exc)})
    if not answer.strip():
        answer = synthesize_answer_from_evidence(user_message, used_chunks, web_sources)
    messages.append({"role": "assistant", "content": answer})
    return finalize_result(user_message, classification, answer, used_chunks, web_sources, tool_history, messages, config)


def run_chat_turn(user_message: str, chat_history: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    classification = classify_question_need(user_message)
    if config.get("no_tools"):
        return run_fallback_evidence_answer(user_message, chat_history, config, classification=classification)
    messages = initial_messages(user_message, chat_history)
    tools = build_tools_schema()
    tool_history: list[dict[str, Any]] = []
    used_chunks: dict[str, dict[str, Any]] = {}
    web_sources: list[dict[str, Any]] = []

    for _ in range(int(config.get("max_agent_steps") or 6)):
        response = ollama_chat(messages, tools, config, stream=False)
        assistant_message = response.get("message") or {"role": "assistant", "content": ""}
        tool_calls = extract_tool_calls(response)
        if tool_calls:
            messages.append(assistant_message)
            for call in tool_calls:
                entry = execute_tool_call(call, config)
                tool_history.append(entry)
                collect_used_chunks_from_tool_result(entry, used_chunks)
                collect_web_sources_from_tool_result(entry, web_sources)
                messages.append(tool_message_from_result(entry, call.get("id")))
            continue

        answer = assistant_message.get("content") or ""
        has_law = any(entry.get("name") == "search_laws" for entry in tool_history)
        has_web = any(entry.get("name") in {"web_search", "official_web_search"} for entry in tool_history)
        if (classification.get("need_law_db") and not has_law) or (classification.get("need_web") and not has_web):
            return run_fallback_evidence_answer(
                user_message,
                chat_history,
                config,
                previous_messages=messages,
                classification=classification,
                tool_history=tool_history,
                used_chunks=used_chunks,
                web_sources=web_sources,
            )
        messages.append({"role": "assistant", "content": answer})
        return finalize_result(user_message, classification, answer, used_chunks, web_sources, tool_history, messages, config)

    messages.append({"role": "user", "content": "请停止调用工具，基于已经返回的工具证据给出最终答案。"})
    response = ollama_chat(messages, None, config, stream=False)
    answer = (response.get("message") or {}).get("content") or ""
    if not answer:
        answer = synthesize_answer_from_evidence(user_message, used_chunks, web_sources)
    messages.append({"role": "assistant", "content": answer})
    return finalize_result(user_message, classification, answer, used_chunks, web_sources, tool_history, messages, config)


def format_used_chunks(used_chunks: Any) -> str:
    if isinstance(used_chunks, dict):
        chunks = used_chunks.values()
    else:
        chunks = used_chunks or []
    chunks = list(chunks)
    if not chunks:
        return "使用本地法规证据：无"
    lines = ["使用本地法规证据："]
    for chunk in chunks:
        lines.append(
            f"- chunk_id={chunk.get('chunk_id') or ''} | {chunk.get('doc_title') or ''} | {chunk.get('article_no') or ''} | 页码 {chunk.get('page_start') or ''}"
        )
    return "\n".join(lines)


def format_web_sources(web_sources: list[dict[str, Any]]) -> str:
    if not web_sources:
        return "使用联网资料：无"
    lines = ["使用联网资料："]
    for source in web_sources:
        lines.append(f"- {source.get('title') or '未命名'} | {source.get('url') or ''}")
    return "\n".join(lines)


def run_interactive(config: dict[str, Any]) -> None:
    print("Web-Augmented 法规 Agent 已启动")
    print(f"LLM：{config['llm_model']}")
    print(f"本地法规库：{Path(config['sqlite_db']).name}")
    print(f"Qdrant：{config['collection']}")
    print(f"Web Provider：{config['web_provider']}")
    print(f"SearXNG：{config['searxng_url']}")
    print("你可以问：")
    print("- 医疗机构发现艾滋病病例后应当如何报告？")
    print("- 这个规定有没有最新官方说法？")
    print("- 结合本地法规和网上公开资料解释一下……")
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
        print(format_used_chunks(result.get("used_chunks") or []))
        print(format_web_sources(result.get("web_sources") or []))
        warnings = result.get("citation_check", {}).get("warnings") or []
        if warnings:
            print("引用检查 warning：")
            for warning in warnings:
                print(f"- {warning}")
        if config.get("debug"):
            print(json.dumps(result.get("tool_history") or [], ensure_ascii=False, indent=2))
        chat_history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result["answer"]},
            ]
        )
        chat_history = trim_chat_history(chat_history)


def build_openai_completion(answer: str, model_name: str, created: int) -> dict[str, Any]:
    return _base_host.build_openai_completion(answer, model_name, created)


def iter_openai_sse(answer: str, model_name: str, created: int):
    yield from _base_host.iter_openai_sse(answer, model_name, created)


def create_fastapi_app(config: dict[str, Any]):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise LawChatHostError(DEPENDENCY_MESSAGE) from exc

    app = FastAPI(title="Web-Augmented Law Agent Host")

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
            "web_provider": config.get("web_provider"),
            "searxng_url": config.get("searxng_url"),
        }

    @app.get("/tools")
    def tools():
        return {"tools": build_tools_schema()}

    @app.post("/chat")
    def chat(request: ChatRequest = Body(...)):
        try:
            return run_chat_turn(message_content_to_text(request.message), normalize_host_history(request.history), config)
        except LawChatHostError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/chat/completions")
    def openai_chat(request: OpenAIChatRequest = Body(...)):
        latest, history = extract_latest_user_message(request.messages or [])
        if not latest:
            answer = "未收到用户问题"
        else:
            try:
                result = run_chat_turn(latest, history, config)
            except LawChatHostError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            answer = result["answer"]
        created = int(time.time())
        model_name = request.model or "law-web-agent-qwen3.6"
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
    path = AGENT_LOG_DIR / f"law_web_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "user_message": user_message,
        "classification": result.get("classification"),
        "answer": result.get("answer"),
        "used_chunks": result.get("used_chunks"),
        "web_sources": result.get("web_sources"),
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
        if config["web_top_k"] < 1:
            raise LawChatHostError("--web-top-k 必须大于 0")
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
