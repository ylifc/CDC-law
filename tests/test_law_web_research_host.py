import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "07_law_web_research_host.py"


def load_module():
    spec = importlib.util.spec_from_file_location("law_web_research_host", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_config(tmp_path: Path) -> dict:
    return {
        "root_dir": Path("/home/ylifc/AI_models_works/baoancdc_law"),
        "output_dir": tmp_path,
        "sqlite_db": tmp_path / "law_rag.db",
        "ollama_url": "http://localhost:11434",
        "llm_model": "qwen3.6:27b",
        "embedding_model": "qwen3-embedding:0.6b",
        "qdrant_url": "http://localhost:6333",
        "collection": "law_chunks_local",
        "top_k": 3,
        "keyword_top_k": 5,
        "semantic_top_k": 5,
        "web_provider": "searxng",
        "searxng_url": "http://127.0.0.1:8080",
        "web_top_k": 3,
        "max_agent_steps": 3,
        "status": None,
        "category": None,
        "doc_title": None,
        "access_level": None,
        "save_log": False,
        "debug": False,
        "no_tools": False,
        "host": "127.0.0.1",
        "port": 8009,
        "ollama_timeout": 60,
        "num_predict": 512,
    }


def test_build_config_and_tools_schema_include_web_tools():
    module = load_module()
    args = argparse.Namespace(
        interactive=False,
        serve=False,
        host="127.0.0.1",
        port=8009,
        llm_model="qwen3.6:27b",
        embedding_model="qwen3-embedding:0.6b",
        ollama_url="http://localhost:11434",
        qdrant_url="http://localhost:6333",
        collection="law_chunks_local",
        web_provider="searxng",
        searxng_url="http://127.0.0.1:8080",
        web_top_k=5,
        top_k=8,
        keyword_top_k=30,
        semantic_top_k=30,
        max_agent_steps=6,
        status=None,
        category=None,
        doc_title=None,
        access_level=None,
        save_log=False,
        debug=False,
        no_tools=False,
    )

    config = module.build_config(args)
    tools = module.build_tools_schema()

    assert config["interactive"] is True
    assert config["port"] == 8009
    assert config["web_provider"] == "searxng"
    assert config["web_top_k"] == 5
    assert "你是一个本地法律法规研究型 Agent" in module.get_system_prompt()
    assert {tool["function"]["name"] for tool in tools} == {
        "search_laws",
        "read_chunk",
        "check_validity",
        "list_categories",
        "web_search",
        "official_web_search",
        "web_fetch",
    }


def test_classify_question_need_routes_law_web_and_official():
    module = load_module()

    result = module.classify_question_need("医疗机构发现艾滋病病例后应当如何报告？有没有最新官方说法？")

    assert result["need_law_db"] is True
    assert result["need_web"] is True
    assert result["need_official_web"] is True
    assert "法规" in result["reason"] or "官方" in result["reason"]


def test_web_search_uses_searxng_site_filter_and_handles_connection_errors(monkeypatch):
    module = load_module()
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.trust_env = True

        def get(self, url, params=None, headers=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            return FakeResponse(
                {
                    "results": [
                        {
                            "title": "国家卫生健康委通知",
                            "url": "https://www.nhc.gov.cn/a.html",
                            "content": "官方通知摘要",
                            "engine": "bing",
                        }
                    ]
                }
            )

    monkeypatch.setattr(module.requests, "Session", FakeSession)

    result = module.web_search("艾滋病 报告", top_k=2, searxng_url="http://127.0.0.1:8080", site_filter="nhc.gov.cn")

    assert result["ok"] is True
    assert result["tool"] == "web_search"
    assert captured["params"]["q"].startswith("site:nhc.gov.cn ")
    assert captured["params"]["format"] == "json"
    assert result["results"][0]["source"] == "bing"
    assert result["results"][0]["snippet"] == "官方通知摘要"

    class BrokenSession(FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):
            raise module.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(module.requests, "Session", BrokenSession)
    failed = module.web_search("艾滋病", searxng_url="http://127.0.0.1:8080")

    assert failed["ok"] is False
    assert "未连接到 SearXNG" in failed["error"]


def test_tavily_search_requires_key_and_normalizes_results(monkeypatch):
    module = load_module()

    missing = module.tavily_search("职业病 最新", top_k=2, api_key=None)
    assert missing["ok"] is False
    assert "TAVILY_API_KEY" in missing["error"]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "职业病政策",
                        "url": "https://www.gov.cn/policy.html",
                        "content": "政策摘要",
                        "score": 0.92,
                    }
                ]
            }

    class FakeSession:
        def __init__(self):
            self.trust_env = True

        def post(self, url, json=None, headers=None, timeout=None):
            assert headers["Authorization"] == "Bearer secret-key"
            assert json["query"] == "职业病 最新"
            return FakeResponse()

    monkeypatch.setattr(module.requests, "Session", FakeSession)

    result = module.tavily_search("职业病 最新", top_k=2, api_key="secret-key")

    assert result["ok"] is True
    assert result["results"][0] == {
        "title": "职业病政策",
        "url": "https://www.gov.cn/policy.html",
        "snippet": "政策摘要",
        "content": "政策摘要",
        "score": 0.92,
    }


def test_web_fetch_rejects_ssrf_and_extracts_text_with_bs4_fallback(monkeypatch):
    module = load_module()

    blocked = module.web_fetch("http://127.0.0.1:8000/private")
    assert blocked["ok"] is False
    assert "禁止访问" in blocked["error"]

    class FakeResponse:
        text = "<html><head><title>政策解读</title></head><body><script>x</script><h1>标题</h1><p>正文内容</p></body></html>"
        content = text.encode("utf-8")

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.trust_env = True

        def get(self, url, headers=None, timeout=None):
            return FakeResponse()

    monkeypatch.setattr(module.requests, "Session", FakeSession)
    monkeypatch.setattr(module, "trafilatura", None)

    result = module.web_fetch("https://www.nhc.gov.cn/policy.html", max_chars=20)

    assert result["ok"] is True
    assert result["tool"] == "web_fetch"
    assert result["title"] == "政策解读"
    assert "正文内容" in result["text"]
    assert result["text_length"] <= 20


def test_official_web_search_merges_official_sites_and_deduplicates(monkeypatch, tmp_path):
    module = load_module()
    config = base_config(tmp_path)
    calls = []

    def fake_web_search(query, top_k=5, searxng_url=None, site_filter=None):
        calls.append(site_filter)
        return {
            "ok": True,
            "results": [
                {
                    "title": f"{site_filter} 结果",
                    "url": "https://www.nhc.gov.cn/a.html" if site_filter != "gov.cn" else "https://www.gov.cn/a.html",
                    "snippet": "摘要",
                    "source": "searxng",
                }
            ],
        }

    monkeypatch.setattr(module, "web_search", fake_web_search)

    result = module.official_web_search("传染病 报告 最新", top_k=5, config=config)

    assert result["ok"] is True
    assert result["tool"] == "official_web_search"
    assert "gov.cn" in calls
    assert "nhc.gov.cn" in calls
    assert result["result_count"] == 2
    assert all(item["source_type"] == "official" for item in result["results"])


def test_validate_web_citations_warns_for_missing_and_unknown_urls():
    module = load_module()
    sources = [{"url": "https://www.nhc.gov.cn/a.html", "title": "A"}]

    missing = module.validate_web_citations("联网公开资料：国家卫健委有相关说明。", sources)
    unknown = module.validate_web_citations("联网公开资料：https://example.com/a.html", sources)
    known = module.validate_web_citations("联网公开资料：https://www.nhc.gov.cn/a.html", sources)

    assert missing["warnings"] == ["答案包含联网公开资料部分，但没有给出 URL"]
    assert unknown["warnings"] == ["回答引用了未检索到的 URL：https://example.com/a.html"]
    assert known["warnings"] == []


def test_run_chat_turn_fallback_calls_law_and_official_web_when_model_skips_tools(monkeypatch, tmp_path):
    module = load_module()
    config = base_config(tmp_path)
    llm_calls = []

    def fake_ollama_chat(messages, tools, config, stream=False):
        llm_calls.append(messages)
        if len(llm_calls) == 1:
            return {"message": {"role": "assistant", "content": "按规定处理即可。"}}
        return {
            "message": {
                "role": "assistant",
                "content": (
                    "结论：应依法报告。\n"
                    "本地法规库依据：1. 《艾滋病防治条例》第三十八条，第12页，chunk_id=c1：应报告。\n"
                    "联网公开资料：1. 标题：国家卫健委通知\n来源：国家卫生健康委员会\nURL：https://www.nhc.gov.cn/a.html\n摘要：最新官方说法。\n"
                    "综合判断：本地和官网一致。\n注意事项：按现行正式文本核验。\n未覆盖或证据不足：无。"
                ),
            }
        }

    monkeypatch.setattr(module, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(
        module,
        "search_laws",
        lambda *args, **kwargs: {
            "ok": True,
            "results": [
                {
                    "chunk_id": "c1",
                    "doc_title": "艾滋病防治条例",
                    "article_no": "第三十八条",
                    "page_start": 12,
                    "content_preview": "应报告。",
                }
            ],
        },
    )
    monkeypatch.setattr(
        module,
        "official_web_search",
        lambda *args, **kwargs: {
            "ok": True,
            "results": [
                {
                    "title": "国家卫健委通知",
                    "url": "https://www.nhc.gov.cn/a.html",
                    "snippet": "最新官方说法。",
                    "source": "国家卫生健康委员会",
                    "source_type": "official",
                }
            ],
        },
    )

    result = module.run_chat_turn("医疗机构发现艾滋病病例后应当如何报告？有没有最新官方说法？", [], config)

    assert result["classification"]["need_law_db"] is True
    assert result["classification"]["need_official_web"] is True
    assert [item["name"] for item in result["tool_history"]] == ["search_laws", "official_web_search"]
    assert result["used_chunk_count"] == 1
    assert result["web_source_count"] == 1
    assert result["citation_check"]["warnings"] == []


def test_create_fastapi_app_routes(monkeypatch, tmp_path):
    module = load_module()
    config = base_config(tmp_path)

    monkeypatch.setattr(
        module,
        "run_chat_turn",
        lambda message, history, config: {
            "question": message,
            "answer": "答复",
            "used_chunks": [],
            "web_sources": [],
            "tool_history": [],
            "citation_check": {},
        },
    )
    app = module.create_fastapi_app(config)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    chat_response = client.post("/chat", json={"message": "你好", "history": []})
    openai_response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "你好"}]})

    assert chat_response.status_code == 200
    assert chat_response.json()["answer"] == "答复"
    assert openai_response.status_code == 200
    assert openai_response.json()["choices"][0]["message"]["content"] == "答复"
