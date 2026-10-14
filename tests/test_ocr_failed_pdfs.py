import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02_ocr_failed_pdfs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ocr_failed_pdfs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_text_from_paddle_result_handles_common_shapes():
    module = load_module()

    assert module.extract_text_from_paddle_result([[[[0, 0]], ("第一条 正文", 0.99)]]) == ["第一条 正文"]
    assert module.extract_text_from_paddle_result([{"rec_texts": ["第二条 内容", ""], "other": 1}]) == ["第二条 内容"]
    assert module.extract_text_from_paddle_result({"rec_texts": ["第三条 内容"]}) == ["第三条 内容"]


def test_clean_ocr_lines_normalizes_markers_and_drops_toc_page():
    module = load_module()
    lines = [
        {"page": 1, "text": "目录"},
        {"page": 1, "text": "第一章 总则"},
        {"page": 1, "text": "第二章 监督"},
        {"page": 1, "text": "第一条 目录行"},
        {"page": 2, "text": "第 一 条  为了 测试"},
        {"page": 2, "text": "  1  "},
        {"page": 2, "text": "第 十 二 条  继续"},
    ]

    cleaned = module.clean_ocr_lines(lines)

    assert {"page": 2, "text": "第一条 为了 测试"} in cleaned
    assert {"page": 2, "text": "第十二条 继续"} in cleaned
    assert all(item["page"] != 1 for item in cleaned)
    assert all(item["text"] != "1" for item in cleaned)


def test_split_ocr_chunks_tracks_chapter_section_pages_and_clause_split():
    module = load_module()
    doc_info = {
        "doc_id": "doc075",
        "doc_title": "测试法规",
        "doc_no": None,
        "issuing_body": None,
        "publish_date": "2019-03-02",
        "effective_date": None,
        "expiry_date": None,
        "status": "未知",
        "scope": None,
        "law_level": None,
        "access_level": "internal",
        "category_path": "传染病防治/艾滋病",
        "category_level_1": "传染病防治",
        "category_level_2": "艾滋病",
        "category_level_3": None,
        "category_level_4": None,
        "pdf_path": Path("/tmp/test.pdf"),
        "relative_pdf": "传染病防治/艾滋病/test.pdf",
    }
    long_clause = "（一）" + ("甲" * 1810) + "（二）乙"
    lines = [
        {"page": 1, "text": "第一章 总则"},
        {"page": 1, "text": "第一节 一般规定"},
        {"page": 1, "text": "第一条 为了测试"},
        {"page": 2, "text": "第二条 " + long_clause},
    ]

    chunks = []
    seen_ids = set()
    for chunk in module.split_ocr_chunks(lines, doc_info):
        chunks.extend(module.chunk_text_by_clauses(chunk, seen_ids))

    assert chunks[0]["chapter"] == "第一章 总则"
    assert chunks[0]["section"] == "第一节 一般规定"
    assert chunks[0]["article_no"] == "第一条"
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 1
    assert chunks[1]["chunk_id"] == "doc075_article_002_clause_001"
    assert chunks[1]["clause_no"] == "（一）"
    assert chunks[1]["content"].startswith("测试法规 第一章 总则 第二条\n（一）")
    assert chunks[2]["chunk_id"] == "doc075_article_002_clause_002"


def test_merge_chunks_adds_text_parse_method_and_records_collision_warning():
    module = load_module()
    original = [{"chunk_id": "doc075_article_001", "content": "old"}]
    ocr = [{"chunk_id": "doc075_article_001", "content": "new", "parse_method": "ocr"}]

    merged, warnings = module.merge_chunks(original, ocr)

    assert merged[0]["parse_method"] == "text"
    assert merged[1]["chunk_id"] == "doc075_article_001_ocrfix_001"
    assert warnings and "doc075_article_001" in warnings[0]


def test_validate_chunks_rejects_bad_status_and_duplicate_ids():
    module = load_module()
    chunk = {
        "chunk_id": "same",
        "doc_id": "doc001",
        "doc_title": "标题",
        "content": "第一条 内容",
        "article_no": "第一条",
        "page_start": 1,
        "category_path": "分类",
        "source_file": "/tmp/test.pdf",
        "status": "乱填",
        "parse_method": "ocr",
    }

    errors = module.validate_chunks([chunk, dict(chunk)])

    assert any("重复" in error for error in errors)
    assert any("status" in error for error in errors)


def test_split_ocr_chunks_falls_back_to_numbered_headings_without_articles():
    module = load_module()
    doc_info = {
        "doc_id": "doc076",
        "doc_title": "医院隔离技术规范03",
        "doc_no": None,
        "issuing_body": None,
        "publish_date": "2009",
        "effective_date": None,
        "expiry_date": None,
        "status": "未知",
        "scope": None,
        "law_level": None,
        "access_level": "internal",
        "category_path": "传染病防治",
        "category_level_1": "传染病防治",
        "category_level_2": None,
        "category_level_3": None,
        "category_level_4": None,
        "pdf_path": Path("/tmp/test.pdf"),
        "relative_pdf": "传染病防治/test.pdf",
    }
    lines = [
        {"page": 1, "text": "医院隔离技术规范"},
        {"page": 2, "text": "1范围"},
        {"page": 2, "text": "本标准规定了医院隔离的管理要求。"},
        {"page": 3, "text": "2规范性引用文件"},
        {"page": 3, "text": "GB19082医用一次性防护服技术要求"},
    ]

    chunks = module.split_ocr_chunks(lines, doc_info)

    assert len(chunks) == 2
    assert chunks[0]["article_no"] == "1"
    assert chunks[0]["content"].startswith("1范围")
    assert chunks[1]["article_no"] == "2"
    assert chunks[1]["page_start"] == 3
