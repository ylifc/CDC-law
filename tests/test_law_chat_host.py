import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_law_chat_host.py"


def load_module():
    spec = importlib.util.spec_from_file_location("law_chat_host", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_law_chunks_db(db_path: Path) -> None:
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
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_title": "艾滋病防治条例",
            "doc_no": None,
            "issuing_body": "国务院",
            "publish_date": "2019-03-02",
            "effective_date": "2019-03-02",
            "expiry_date": None,
            "status": "现行有效",
            "scope": None,
            "law_level": "行政法规",
            "category_path": "传染病防治/艾滋病",
            "category_level_1": "传染病防治",
            "category_level_2": "艾滋病",
            "category_level_3": None,
            "category_level_4": None,
            "chapter": "第五章",
            "section": None,
            "article_no": "第三十八条",
            "clause_no": None,
            "content": "第三十八条 医疗机构发现艾滋病病例后应当依法报告。",
            "source_file": "/tmp/a.pdf",
            "relative_source_file": "传染病防治/艾滋病/a.pdf",
            "page_start": 12,
            "page_end": 13,
            "access_level": "internal",
            "parse_method": "text",
            "content_for_embedding": "embedding",
            "content_length": 28,
            "created_at": "2026-07-03T00:00:00Z",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "doc_title": "职业病防治法",
            "doc_no": None,
            "issuing_body": None,
            "publish_date": "2018-12-29",
            "effective_date": "2018-12-29",
            "expiry_date": None,
            "status": "未知",
            "scope": None,
            "law_level": "法律",
            "category_path": "职业卫生",
            "category_level_1": "职业卫生",
            "category_level_2": None,
            "category_level_3": None,
            "category_level_4": None,
            "chapter": None,
            "section": None,
            "article_no": "第一条",
            "clause_no": None,
            "content": "职业病报告要求。",
            "source_file": "/tmp/b.pdf",
            "relative_source_file": "职业卫生/b.pdf",
            "page_start": 1,
            "page_end": 1,
            "access_level": "internal",
            "parse_method": "text",
            "content_for_embedding": "embedding",
            "content_length": 8,
            "created_at": "2026-07-03T00:00:00Z",
        },
    ]
    columns = list(rows[0].keys())
    for row in rows:
        conn.execute(
            f"INSERT INTO law_chunks ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})",
            [row[column] for column in columns],
        )
    conn.commit()
    conn.close()


def base_config(db_path: Path) -> dict:
    return {
        "root_dir": Path("/home/ylifc/AI_models_works/baoancdc_law"),
        "output_dir": db_path.parent,
        "sqlite_db": db_path,
        "ollama_url": "http://localhost:11434",
        "llm_model": "qwen3.6:27b",
        "embedding_model": "qwen3-embedding:0.6b",
        "qdrant_url": "http://localhost:6333",
        "collection": "law_chunks_local",
        "top_k": 2,
        "keyword_top_k": 5,
        "semantic_top_k": 5,
        "max_agent_steps": 3,
        "status": None,
        "category": None,
        "doc_title": None,
        "access_level": None,
        "save_log": False,
        "debug": False,
        "no_tools": False,
        "host": "127.0.0.1",
        "port": 8008,
        "web_enabled": True,
        "web_timeout": 12,
        "official_updates_dir": db_path.parent / "official_updates_pending",
        "official_download_enabled": True,
        "download_timeout": 20,
        "max_download_bytes": 50_000_000,
    }


def test_build_config_defaults_to_interactive():
    module = load_module()
    args = argparse.Namespace(
        interactive=False,
        serve=False,
        host="127.0.0.1",
        port=8008,
        llm_model="qwen3.6:27b",
        embedding_model="qwen3-embedding:0.6b",
        ollama_url="http://localhost:11434",
        qdrant_url="http://localhost:6333",
        collection="law_chunks_local",
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
        no_web=False,
        web_timeout=12,
        no_official_download=False,
        official_updates_dir=None,
        download_timeout=20,
        max_download_bytes=50_000_000,
    )

    config = module.build_config(args)

    assert config["interactive"] is True
    assert config["serve"] is False
    assert config["llm_model"] == "qwen3.6:27b"
    assert config["sqlite_db"].name == "law_rag.db"
    assert config["web_enabled"] is True
    assert config["official_download_enabled"] is True
    assert config["official_updates_dir"].name == "official_updates_pending"
    assert config["download_timeout"] == 20
    assert config["max_download_bytes"] == 50_000_000


def test_system_prompt_tools_schema_and_law_question_detection():
    module = load_module()

    prompt = module.get_system_prompt()
    tools = module.build_tools_schema()

    assert "必须先调用 search_laws" in prompt
    assert "official_web_search" in prompt
    assert "不要为了套模板重复空泛段落" in prompt
    assert {tool["function"]["name"] for tool in tools} == {
        "search_laws",
        "read_chunk",
        "check_validity",
        "list_categories",
        "official_web_search",
    }
    assert module.is_law_related_question("医疗机构发现艾滋病病例后应当如何报告？") is True
    assert module.is_law_related_question("你好，今天心情如何？") is False


def test_china_cdc_official_sources_cover_laws_and_manuals():
    module = load_module()

    china_cdc_sources = [
        source for source in module.OFFICIAL_INDEX_SOURCES if source["source"] == "中国疾病预防控制中心"
    ]
    urls = china_cdc_sources[0]["urls"]

    assert "https://www.chinacdc.cn/jkyj/" in urls
    assert "https://www.chinacdc.cn/jkyj/yyyjk2/jswj13949/" in urls
    assert "https://www.chinacdc.cn/jkyj/fsws/fgbz/fg/gfxwj/" in urls
    assert "https://ivdc.chinacdc.cn/cnic/zyzx/syssc/" in urls
    assert module.is_allowed_official_url("https://ivdc.chinacdc.cn/cnic/zyzx/syssc/201605/t20160520_129703.htm")
    assert module.is_allowed_official_url("https://www.chinacdc.cn/jkyj/fsws/fgbz/fg/gfxwj/")
    assert not module.is_allowed_official_url("https://notchinacdc.cn/fake")


def test_nhc_and_ndcpa_sources_cover_health_rules_and_disease_classification():
    module = load_module()

    nhc_sources = [source for source in module.OFFICIAL_INDEX_SOURCES if source["source"] == "国家卫生健康委员会"]
    ndcpa_sources = [source for source in module.OFFICIAL_INDEX_SOURCES if source["source"] == "国家疾病预防控制局"]
    nhc_urls = nhc_sources[0]["urls"]
    ndcpa_urls = ndcpa_sources[0]["urls"]

    assert "https://www.nhc.gov.cn/" in nhc_urls
    assert "https://www.nhc.gov.cn/zwgkzt/pzcfg/list.shtml" in nhc_urls
    assert "https://www.nhc.gov.cn/zhengce/" in nhc_urls
    assert "https://www.nhc.gov.cn/wjw/gfxwj/list.shtml" in nhc_urls
    assert "https://www.nhc.gov.cn/wjw/zcjd/list.shtml" in nhc_urls
    assert "https://www.nhc.gov.cn/ylyjs/gzdt/list.shtml" in nhc_urls
    assert "https://www.nhc.gov.cn/ylyjs/zcwj/list.shtml" in nhc_urls
    assert "https://www.ndcpa.gov.cn/jbkzzx/c1000372/common/list.html" in ndcpa_urls
    assert "https://www.ndcpa.gov.cn/jbkzzx/c100041/common/list.html" in ndcpa_urls
    assert "https://www.ndcpa.gov.cn/jbkzzx/c100013/common/list.html" in ndcpa_urls


def test_gov_cn_official_links_are_allowed_but_third_party_links_are_rejected():
    module = load_module()

    assert module.is_allowed_official_url("https://www.nhc.gov.cn/wjw/gfxwj/list.shtml")
    assert module.is_allowed_official_url("https://wsjkw.henan.gov.cn/zwgk/zcfg/")
    assert module.is_allowed_official_url("https://www.gov.cn/zhengce/zhengceku/")
    assert not module.is_allowed_official_url("https://gov.cn.example.com/fake.pdf")
    assert not module.is_allowed_official_url("https://example.com/gov.cn/fake.pdf")


def test_discover_download_urls_accepts_gov_cn_pdf_links():
    module = load_module()

    item = {
        "title": "卫生健康规定",
        "url": "https://www.nhc.gov.cn/policy.html",
        "download_urls": ["https://wsjkw.henan.gov.cn/files/health_rule.pdf"],
    }

    class FakeSession:
        pass

    urls = module.discover_download_urls(item, FakeSession(), timeout=1)

    assert urls == ["https://wsjkw.henan.gov.cn/files/health_rule.pdf"]


def test_read_chunk_check_validity_and_list_categories(tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    create_law_chunks_db(db_path)
    config = base_config(db_path)

    chunk = module.read_chunk("c1", config=config)
    validity = module.check_validity(chunk_id="c1", config=config)
    categories = module.list_categories(limit=10, config=config)

    assert chunk["ok"] is True
    assert chunk["chunk"]["content"].startswith("第三十八条")
    assert validity["ok"] is True
    assert validity["status"] == "现行有效"
    assert "传染病防治/艾滋病" in [item["category_path"] for item in categories["categories"]]


def test_search_laws_returns_compact_previews(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)

    def fake_hybrid_search(**kwargs):
        return [
            {
                "rank": 1,
                "chunk_id": "c1",
                "doc_title": "艾滋病防治条例",
                "doc_no": None,
                "article_no": "第三十八条",
                "chapter": "第五章",
                "page_start": 12,
                "page_end": 13,
                "status": "现行有效",
                "category_path": "传染病防治/艾滋病",
                "score": 0.123,
                "content": "甲" * 700,
            }
        ]

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = module.search_laws("艾滋病报告", top_k=1, config=config)

    assert result["ok"] is True
    assert result["results"][0]["chunk_id"] == "c1"
    assert len(result["results"][0]["content_preview"]) == 500


def test_official_web_search_uses_whitelist_sources(monkeypatch, tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    captured = {"trust_env": None, "urls": []}

    class FakeResponse:
        def __init__(self, payload=None, text=""):
            self._payload = payload
            self.text = text

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.trust_env = False

        def get(self, url, params=None, headers=None, timeout=None):
            captured["trust_env"] = self.trust_env
            captured["urls"].append(url)
            if "flk.npc.gov.cn/api" in url:
                return FakeResponse(
                    payload={
                        "success": True,
                        "result": {
                            "data": [
                                {
                                    "title": "中华人民共和国传染病防治法",
                                    "office": "全国人民代表大会常务委员会",
                                    "publish": "2025-01-01 00:00:00",
                                    "type": "法律",
                                    "status": "1",
                                    "url": "https://flk.npc.gov.cn/detail2.html?abc",
                                }
                            ]
                        },
                    }
                )
            return FakeResponse(
                text="""
                <html><body>
                  <a href="/fzs/s7852/202601/abc.shtml">关于印发传染病信息报告管理规范的通知</a><span>2026-01-02</span>
                  <a href="https://example.com/not-official.html">非官方网站结果</a>
                </body></html>
                """
            )

    monkeypatch.setattr(module.requests, "Session", FakeSession)

    result = module.official_web_search("传染病信息报告管理规范", limit=5, config=config)

    assert result["ok"] is True
    assert captured["trust_env"] is True
    urls = [item["url"] for item in result["results"]]
    assert "https://flk.npc.gov.cn/detail2.html?abc" in urls
    assert "https://www.nhc.gov.cn/fzs/s7852/202601/abc.shtml" in urls
    assert all("example.com" not in url for url in urls)
    assert result["results"][0]["source"]
    assert result["results"][0]["title"]


def test_official_web_search_reads_direct_nhc_url_with_reader_fallback(monkeypatch, tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    nhc_url = "https://www.nhc.gov.cn/ylyjs/gzdt/202603/dec2534f394540c3b3d8c87fb3cad26d.shtml"
    captured_urls = []

    class FakeResponse:
        def __init__(self, payload=None):
            self._payload = payload
            self.text = json.dumps(payload or {}, ensure_ascii=False)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.trust_env = False

        def get(self, url, params=None, headers=None, timeout=None):
            captured_urls.append(url)
            if url.startswith(module.OFFICIAL_READER_PREFIX):
                return FakeResponse(
                    {
                        "code": 200,
                        "data": {
                            "title": "将基孔肯雅热和发热伴血小板减少综合征纳入乙类传染病管理的政策解读",
                            "url": nhc_url,
                            "content": (
                                "基孔肯雅热是一种由基孔肯雅病毒引起的急性传染病。"
                                "自2026年4月1日起将基孔肯雅热纳入乙类传染病管理。"
                                "发热伴血小板减少综合征也纳入乙类传染病管理。"
                            ),
                            "metadata": {"PubDate": "2026-04-02 10:28:31"},
                        },
                    }
                )
            raise module.requests.exceptions.SSLError("tls eof")

    monkeypatch.setattr(module.requests, "Session", FakeSession)

    result = module.official_web_search(f"请查看这个国家规定页面：{nhc_url}", limit=1, config=config)

    assert result["ok"] is True
    assert result["results"][0]["url"] == nhc_url
    assert result["results"][0]["source"] == "国家卫生健康委员会"
    assert result["results"][0]["fetched_via"] == "reader_fallback"
    assert "基孔肯雅热" in result["results"][0]["snippet"]
    assert "发热伴血小板减少综合征" in result["results"][0]["content_preview"]
    assert result["results"][0]["date"] == "2026-04-02"
    assert any(url.startswith(module.OFFICIAL_READER_PREFIX) for url in captured_urls)
    assert any("官网原站抓取失败" in warning for warning in result["warnings"])


def test_compare_local_and_official_flags_official_newer():
    module = load_module()
    local_result = {
        "ok": True,
        "results": [
            {
                "chunk_id": "c1",
                "doc_title": "传染病信息报告管理规范",
                "publish_date": "2018-01-01",
                "effective_date": "2018-02-01",
                "status": "现行有效",
            }
        ],
    }
    web_result = {
        "ok": True,
        "results": [
            {
                "source": "国家疾病预防控制局",
                "title": "传染病信息报告管理规范",
                "url": "https://www.ndcpa.gov.cn/policy.html",
                "date": "2026-01-02",
            }
        ],
    }

    comparison = module.compare_local_and_official(local_result, web_result)

    assert comparison["status"] == "official_newer"
    assert comparison["local_items"][0]["chunk_id"] == "c1"
    assert comparison["official_items"][0]["url"] == "https://www.ndcpa.gov.cn/policy.html"
    assert "官网日期晚于本地日期" in comparison["reason"]


def test_compare_local_and_official_handles_missing_and_unknown_dates():
    module = load_module()

    missing = module.compare_local_and_official(
        {"ok": True, "results": []},
        {"ok": True, "results": [{"title": "新规", "url": "https://www.gov.cn/a.html", "date": "2026-01-01"}]},
    )
    unknown = module.compare_local_and_official(
        {"ok": True, "results": [{"chunk_id": "c1", "doc_title": "旧规"}]},
        {"ok": True, "results": [{"title": "旧规", "url": "https://www.gov.cn/a.html"}]},
    )

    assert missing["status"] == "local_missing"
    assert unknown["status"] == "date_unknown"


def test_download_official_update_candidate_saves_pdf_and_metadata(monkeypatch, tmp_path):
    module = load_module()
    config = base_config(tmp_path / "law_rag.db")
    config["official_updates_dir"] = tmp_path / "official_updates_pending"
    comparison = {
        "status": "official_newer",
        "reason": "官网日期晚于本地日期：官网 2026-01-02，本地 2018-01-01。",
        "local_items": [{"chunk_id": "c1", "doc_title": "传染病信息报告管理规范", "date": "2018-01-01"}],
        "official_items": [{"title": "传染病信息报告管理规范", "url": "https://www.ndcpa.gov.cn/policy.pdf", "date": "2026-01-02"}],
    }
    official_item = {
        "source": "国家疾病预防控制局",
        "title": "传染病信息报告管理规范",
        "url": "https://www.ndcpa.gov.cn/policy.html",
        "date": "2026-01-02",
        "download_urls": ["https://www.ndcpa.gov.cn/policy.pdf"],
    }

    class FakeResponse:
        headers = {"Content-Type": "application/pdf"}
        text = ""
        content = b"%PDF-1.7 fake"

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=65536):
            yield self.content

    class FakeSession:
        def __init__(self):
            self.trust_env = False

        def get(self, url, headers=None, timeout=None, stream=False):
            return FakeResponse()

    monkeypatch.setattr(module.requests, "Session", FakeSession)

    records = module.download_official_update_candidates([official_item], comparison, config)

    assert len(records) == 1
    record = records[0]
    saved_file = Path(record["file_path"])
    metadata_file = Path(record["metadata_path"])
    assert saved_file.exists()
    assert saved_file.parent.name == "pdf"
    assert saved_file.read_bytes().startswith(b"%PDF")
    assert metadata_file.exists()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["title"] == "传染病信息报告管理规范"
    assert metadata["download_url"] == "https://www.ndcpa.gov.cn/policy.pdf"
    assert metadata["comparison_status"] == "official_newer"


def test_download_official_update_candidate_saves_html_with_reader_fallback(monkeypatch, tmp_path):
    module = load_module()
    config = base_config(tmp_path / "law_rag.db")
    config["official_updates_dir"] = tmp_path / "official_updates_pending"
    nhc_url = "https://www.nhc.gov.cn/ylyjs/gzdt/202603/dec2534f394540c3b3d8c87fb3cad26d.shtml"
    comparison = {
        "status": "local_missing",
        "reason": "本地库未检索到对应依据，官网结果作为待核对候选。",
        "local_items": [],
        "official_items": [{"title": "政策解读", "url": nhc_url, "date": "2026-04-02"}],
    }
    official_item = {
        "source": "国家卫生健康委员会",
        "title": "政策解读",
        "url": nhc_url,
        "date": "2026-04-02",
    }

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload, ensure_ascii=False)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.trust_env = False

        def get(self, url, headers=None, timeout=None, stream=False):
            if url.startswith(module.OFFICIAL_READER_PREFIX):
                return FakeResponse(
                    {
                        "code": 200,
                        "data": {
                            "title": "政策解读",
                            "content": "自2026年4月1日起将基孔肯雅热和发热伴血小板减少综合征纳入乙类传染病管理。",
                            "metadata": {"PubDate": "2026-04-02 10:28:31"},
                        },
                    }
                )
            raise module.requests.exceptions.SSLError("tls eof")

    monkeypatch.setattr(module.requests, "Session", FakeSession)

    records = module.download_official_update_candidates([official_item], comparison, config)

    assert len(records) == 1
    record = records[0]
    saved_file = Path(record["file_path"])
    metadata_file = Path(record["metadata_path"])
    assert saved_file.exists()
    assert saved_file.parent.name == "html"
    assert "乙类传染病管理" in saved_file.read_text(encoding="utf-8")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["fetched_via"] == "reader_fallback"
    assert metadata["download_url"] == nhc_url


def test_extract_tool_calls_accepts_ollama_shapes():
    module = load_module()
    response = {
        "message": {
            "tool_calls": [
                {
                    "function": {
                        "name": "search_laws",
                        "arguments": "{\"query\":\"艾滋病报告\",\"top_k\":2}",
                    }
                }
            ]
        }
    }

    calls = module.extract_tool_calls(response)

    assert calls == [{"name": "search_laws", "arguments": {"query": "艾滋病报告", "top_k": 2}}]


def test_ollama_chat_sends_tools_and_generation_limits(monkeypatch, tmp_path):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    config["num_predict"] = 256
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"role": "assistant", "content": "ok"}}

    class FakeSession:
        def __init__(self):
            self.trust_env = True

        def post(self, url, json, timeout):
            captured["url"] = url
            captured["body"] = json
            captured["timeout"] = timeout
            captured["trust_env"] = self.trust_env
            return FakeResponse()

    monkeypatch.setattr(module.requests, "Session", FakeSession)

    response = module.ollama_chat(
        [{"role": "user", "content": "你好"}],
        [{"function": {"name": "search_laws"}}],
        config,
    )

    assert response["message"]["content"] == "ok"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["tools"][0]["function"]["name"] == "search_laws"
    assert captured["body"]["options"]["temperature"] == 0.1
    assert captured["body"]["options"]["num_predict"] == 256
    assert captured["trust_env"] is False


def test_run_chat_turn_fallback_searches_law_database_when_model_skips_tool(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    calls = []

    def fake_ollama_chat(messages, tools, config, stream=False):
        calls.append({"messages": messages, "tools": tools})
        if len(calls) == 1:
            return {"message": {"role": "assistant", "content": "可以，按规定处理。"}}
        return {
            "message": {
                "role": "assistant",
                "content": "结论：应依法报告。\n依据：1. 《艾滋病防治条例》第三十八条，第12-13页，chunk_id=c1：应依法报告。\n适用条件：医疗机构发现病例。\n注意事项：按制度执行。\n未覆盖或证据不足：无。",
            }
        }

    def fake_search_laws(*args, **kwargs):
        return {
            "ok": True,
            "results": [
                {
                    "chunk_id": "c1",
                    "doc_title": "艾滋病防治条例",
                    "article_no": "第三十八条",
                    "page_start": 12,
                    "page_end": 13,
                    "content_preview": "应依法报告。",
                }
            ],
        }

    monkeypatch.setattr(module, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(module, "search_laws", fake_search_laws)

    result = module.run_chat_turn("医疗机构发现艾滋病病例后应当如何报告？", [], config)

    assert "chunk_id=c1" in result["answer"]
    assert "c1" in result["used_chunks"]
    assert result["tool_history"][0]["name"] == "search_laws"
    assert result["citation_check"]["warnings"] == []
    assert len(calls) == 2


def test_fallback_evidence_answer_synthesizes_when_llm_returns_empty(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)

    monkeypatch.setattr(module, "ollama_chat", lambda *args, **kwargs: {"message": {"role": "assistant", "content": ""}})
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
                    "page_end": 13,
                    "content_preview": "医疗机构发现艾滋病病例后应当依法报告。",
                }
            ],
        },
    )

    result = module.run_fallback_evidence_answer("医疗机构发现艾滋病病例后应当如何报告？", [], config)

    assert "chunk_id=c1" in result["answer"]
    assert "《艾滋病防治条例》第三十八条，第12-13页" in result["answer"]
    assert result["citation_check"]["warnings"] == []


def test_fallback_evidence_answer_synthesizes_when_llm_errors(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)

    def raise_ollama(*args, **kwargs):
        raise module.LawChatHostError("Ollama down")

    monkeypatch.setattr(module, "ollama_chat", raise_ollama)
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
                    "page_end": 13,
                    "content_preview": "医疗机构发现艾滋病病例后应当依法报告。",
                }
            ],
        },
    )

    result = module.run_fallback_evidence_answer("医疗机构发现艾滋病病例后应当如何报告？", [], config)

    assert "chunk_id=c1" in result["answer"]
    assert result["citation_check"]["warnings"] == ["LLM 生成失败，已使用检索证据合成保守答案：Ollama down"]


def test_fallback_uses_official_web_when_local_has_no_results(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    config["official_download_enabled"] = False

    monkeypatch.setattr(module, "ollama_chat", lambda *args, **kwargs: {"message": {"role": "assistant", "content": ""}})
    monkeypatch.setattr(module, "search_laws", lambda *args, **kwargs: {"ok": True, "results": []})
    monkeypatch.setattr(
        module,
        "official_web_search",
        lambda *args, **kwargs: {
            "ok": True,
            "results": [
                {
                    "source": "国家疾病预防控制局",
                    "title": "关于印发传染病信息报告管理规范的通知",
                    "url": "https://www.ndcpa.gov.cn/jbkzzx/c100012/common/content.html",
                    "date": "2026-01-02",
                    "snippet": "国家疾控局发布传染病信息报告管理规范。",
                }
            ],
            "warnings": [],
        },
    )

    result = module.run_fallback_evidence_answer("最新国家疾控局传染病报告规范是什么？", [], config)

    assert [item["name"] for item in result["tool_history"]] == ["search_laws", "official_web_search"]
    assert "https://www.ndcpa.gov.cn/jbkzzx/c100012/common/content.html" in result["answer"]
    assert "https://www.ndcpa.gov.cn/jbkzzx/c100012/common/content.html" in result["used_web_sources"]
    assert result["citation_check"]["warnings"] == []


def test_fallback_compares_and_records_downloaded_official_update(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    config["official_updates_dir"] = tmp_path / "official_updates_pending"

    monkeypatch.setattr(module, "ollama_chat", lambda *args, **kwargs: {"message": {"role": "assistant", "content": ""}})
    monkeypatch.setattr(
        module,
        "search_laws",
        lambda *args, **kwargs: {
            "ok": True,
            "results": [
                {
                    "chunk_id": "c1",
                    "doc_title": "传染病信息报告管理规范",
                    "publish_date": "2018-01-01",
                    "effective_date": "2018-02-01",
                    "status": "现行有效",
                    "content_preview": "旧版报告规范。",
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
                    "source": "国家疾病预防控制局",
                    "title": "传染病信息报告管理规范",
                    "url": "https://www.ndcpa.gov.cn/policy.html",
                    "date": "2026-01-02",
                    "snippet": "新版报告规范。",
                    "download_urls": ["https://www.ndcpa.gov.cn/policy.pdf"],
                }
            ],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        module,
        "download_official_update_candidates",
        lambda items, comparison, config: [
            {
                "title": items[0]["title"],
                "source": items[0]["source"],
                "url": items[0]["url"],
                "download_url": "https://www.ndcpa.gov.cn/policy.pdf",
                "file_path": str(config["official_updates_dir"] / "pdf" / "policy.pdf"),
                "metadata_path": str(config["official_updates_dir"] / "metadata" / "policy.json"),
                "comparison_status": comparison["status"],
            }
        ],
    )

    result = module.run_fallback_evidence_answer("最新国家疾控局传染病信息报告管理规范是什么？", [], config)

    assert result["evidence_comparison"]["status"] == "official_newer"
    assert result["downloaded_official_updates"][0]["file_path"].endswith("policy.pdf")
    assert "官网日期晚于本地日期" in result["answer"]
    assert "_rag_output/official_updates_pending" in result["answer"] or "official_updates_pending" in result["answer"]


def test_run_chat_turn_executes_tool_calls_then_final_answer(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    config["max_agent_steps"] = 2
    call_count = {"value": 0}

    def fake_ollama_chat(messages, tools, config, stream=False):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search_laws", "arguments": {"query": "艾滋病报告"}}}
                    ],
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "结论：应报告。\n依据：1. 《艾滋病防治条例》第三十八条，第12页，chunk_id=c1：应报告。\n适用条件：发现病例。\n注意事项：及时。\n未覆盖或证据不足：无。",
            }
        }

    monkeypatch.setattr(module, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(
        module,
        "search_laws",
        lambda *args, **kwargs: {
            "ok": True,
            "results": [{"chunk_id": "c1", "doc_title": "艾滋病防治条例", "content_preview": "应报告。"}],
        },
    )

    result = module.run_chat_turn("艾滋病报告要求", [], config)

    assert call_count["value"] == 2
    assert result["tool_history"][0]["name"] == "search_laws"
    assert "c1" in result["used_chunks"]
    assert result["citation_check"]["warnings"] == []


def test_run_chat_turn_executes_official_web_search_tool(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    config["official_download_enabled"] = False
    config["max_agent_steps"] = 2
    call_count = {"value": 0}

    def fake_ollama_chat(messages, tools, config, stream=False):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search_laws", "arguments": {"query": "最新传染病报告规范"}}},
                        {
                            "function": {
                                "name": "official_web_search",
                                "arguments": {"query": "最新传染病报告规范", "limit": 2},
                            }
                        },
                    ],
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "最新要求以国家疾控局官网为准，依据：国家疾病预防控制局《传染病信息报告管理规范》，https://www.ndcpa.gov.cn/policy.html",
            }
        }

    monkeypatch.setattr(module, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(module, "search_laws", lambda *args, **kwargs: {"ok": True, "results": []})
    monkeypatch.setattr(
        module,
        "official_web_search",
        lambda *args, **kwargs: {
            "ok": True,
            "results": [
                {
                    "source": "国家疾病预防控制局",
                    "title": "传染病信息报告管理规范",
                    "url": "https://www.ndcpa.gov.cn/policy.html",
                    "date": "2026-01-02",
                    "snippet": "关于印发传染病信息报告管理规范的通知",
                }
            ],
            "warnings": [],
        },
    )

    result = module.run_chat_turn("最新传染病报告规范是什么？", [], config)

    assert [item["name"] for item in result["tool_history"][:2]] == ["search_laws", "official_web_search"]
    assert "https://www.ndcpa.gov.cn/policy.html" in result["used_web_sources"]
    assert result["citation_check"]["warnings"] == []


def test_citation_validation_warns_for_missing_and_unknown_chunks():
    module = load_module()

    assert module.extract_cited_chunk_ids("依据 chunk_id=c1；另见 chunk_id: c2。") == ["c1", "c2"]

    missing = module.validate_answer_citations("结论：应报告。", {"c1": {}}, law_related=True)
    unknown = module.validate_answer_citations("依据：chunk_id=c9", {"c1": {}}, law_related=True)
    no_evidence = module.validate_answer_citations("未检索到明确依据", {}, law_related=True)

    assert missing["warnings"] == ["法规相关回答未引用任何 chunk_id 或官方 URL"]
    assert unknown["warnings"] == ["回答引用了未检索过的 chunk_id：c9"]
    assert no_evidence["warnings"] == []


def test_create_fastapi_app_routes(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)

    monkeypatch.setattr(module, "build_tools_schema", lambda: [{"function": {"name": "search_laws"}}])

    def fake_run_chat_turn(message, history, config):
        return {"answer": "答复", "used_chunks": {}, "tool_history": [], "citation_check": {}, "messages": []}

    monkeypatch.setattr(module, "run_chat_turn", fake_run_chat_turn)
    app = module.create_fastapi_app(config)
    routes = {route.path for route in app.routes}

    assert {"/health", "/tools", "/chat", "/v1/chat/completions"}.issubset(routes)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    chat_response = client.post("/chat", json={"message": "你好", "history": []})
    openai_response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )

    assert chat_response.status_code == 200
    assert chat_response.json()["answer"] == "答复"
    assert openai_response.status_code == 200
    assert openai_response.json()["choices"][0]["message"]["content"] == "答复"


def test_openai_endpoint_normalizes_content_parts_and_history(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)
    captured = {}

    def fake_run_chat_turn(message, history, config):
        captured["message"] = message
        captured["history"] = history
        return {"answer": "答复", "used_chunks": {}, "tool_history": [], "citation_check": {}, "messages": []}

    monkeypatch.setattr(module, "run_chat_turn", fake_run_chat_turn)
    app = module.create_fastapi_app(config)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "系统提示不进入历史"},
                {"role": "user", "content": [{"type": "text", "text": "上一轮问题"}]},
                {"role": "assistant", "content": None},
                {"role": "assistant", "content": "上一轮回答"},
                {"role": "tool", "content": "工具结果不进入历史"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "职业暴露感染艾滋病后"},
                        {"type": "image_url", "image_url": {"url": "file://ignored.png"}},
                        {"type": "text", "text": "应在多久报告？"},
                    ],
                },
            ]
        },
    )

    assert response.status_code == 200
    assert captured["message"] == "职业暴露感染艾滋病后应在多久报告？"
    assert captured["history"] == [
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]


def test_openai_endpoint_returns_502_for_host_errors(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)

    def fake_run_chat_turn(message, history, config):
        raise module.LawChatHostError("Ollama 返回 400：content must be string")

    monkeypatch.setattr(module, "run_chat_turn", fake_run_chat_turn)
    app = module.create_fastapi_app(config)

    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "你好"}]})

    assert response.status_code == 502
    assert "content must be string" in response.json()["detail"]


def test_openai_endpoint_streams_sse_when_requested(tmp_path, monkeypatch):
    module = load_module()
    db_path = tmp_path / "law_rag.db"
    config = base_config(db_path)

    def fake_run_chat_turn(message, history, config):
        return {"answer": "流式答复", "used_chunks": {}, "tool_history": [], "citation_check": {}, "messages": []}

    monkeypatch.setattr(module, "run_chat_turn", fake_run_chat_turn)
    app = module.create_fastapi_app(config)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object": "chat.completion.chunk"' in response.text
    assert '"delta": {"content": "流式答复"}' in response.text
    assert "data: [DONE]" in response.text
