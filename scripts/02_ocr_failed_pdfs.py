#!/usr/bin/env python3
"""
扫描版 PDF OCR 补救脚本。

只读取第一轮 parse_report.json 中“扫描版 / 抽取不到文本 / 需要 OCR”的失败 PDF，
使用 PyMuPDF 渲染页面、PaddleOCR 识别文本，并按法规结构生成 OCR chunk。
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm


# Paddle 在部分 CPU 环境的默认后端会走 oneDNN/PIR 组合；禁用后端优化更稳。
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


ROOT_DIR = Path(os.environ.get("CDC_LAW_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
INPUT_PARSE_REPORT = ROOT_DIR / "_rag_output" / "parse_report.json"
ORIGINAL_CHUNKS = ROOT_DIR / "_rag_output" / "chunks.jsonl"
OUTPUT_OCR_CHUNKS = ROOT_DIR / "_rag_output" / "ocr_chunks.jsonl"
OUTPUT_MERGED_CHUNKS = ROOT_DIR / "_rag_output" / "chunks_merged_after_ocr.jsonl"
OUTPUT_OCR_REPORT = ROOT_DIR / "_rag_output" / "ocr_report.json"
OCR_CACHE_DIR = ROOT_DIR / "_rag_output" / "ocr_cache"
OCR_REVIEW_DIR = ROOT_DIR / "_rag_output" / "ocr_review"

OCR_ERROR_KEYWORDS = ("抽取不到文本", "扫描版", "需要 OCR", "需要OCR", "OCR")

CN_NUM = r"[一二三四五六七八九十百千万零〇0-9]+"
CHAPTER_RE = re.compile(rf"^第{CN_NUM}章\s*.*")
SECTION_RE = re.compile(rf"^第{CN_NUM}节\s*.*")
ARTICLE_RE = re.compile(rf"^(第{CN_NUM}条)")
CLAUSE_RE = re.compile(r"^（[一二三四五六七八九十百千万零〇0-9]+）")
CLAUSE_SPLIT_RE = re.compile(r"（[一二三四五六七八九十百千万零〇0-9]+）")
ITEM_RE = re.compile(r"^[0-9]+[\.．、]")
NUMERIC_DOTTED_HEADING_RE = re.compile(r"^([1-9]\d*(?:\.\d+)+)\s*(.*)$")
NUMERIC_TEXT_HEADING_RE = re.compile(r"^([1-9]\d*)\s*([\.．、]?\s*[\u4e00-\u9fff].*)$")
CHINESE_HEADING_RE = re.compile(r"^([一二三四五六七八九十]+[、．.])\s*(.+)$")
APPENDIX_HEADING_RE = re.compile(r"^(附件\s*\d*|附录\s*[A-ZＡ-Ｚ一二三四五六七八九十0-9]*)\s*(.*)$")
PAGE_NUM_RE = re.compile(r"^\s*[-]?\d+[-–—]?\s*$")
LAW_MARKER_SPACE_RE = re.compile(
    r"第\s*([一二三四五六七八九十百千万零〇0-9](?:\s*[一二三四五六七八九十百千万零〇0-9])*)\s*(条|章|节)"
)
DATE_RE = re.compile(
    r"^\s*(\d{4})\s*(?:[./_\-年])\s*(\d{1,2})\s*(?:[./_\-月])\s*(\d{1,2})\s*日?\s*$"
)

VALID_STATUSES = {"现行有效", "已废止", "部分有效", "未知"}
VALID_PARSE_METHODS = {"text", "ocr"}


class PaddleDependencyError(RuntimeError):
    """Raised when PaddleOCR cannot be imported or initialized."""


@dataclass
class ArticleState:
    chapter: Optional[str] = None
    section: Optional[str] = None
    article_no: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    lines: List[str] = None

    def __post_init__(self) -> None:
        if self.lines is None:
            self.lines = []


def normalize_date(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or value.lower() in {"none", "null", "nan"}:
        return None

    match = DATE_RE.match(value)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    for sep in ("-", "/", "_", "."):
        if sep in value:
            parts = [part.strip() for part in value.split(sep)]
            if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
                year, month, day = parts[:3]
                if len(year) == 4:
                    return f"{year}-{int(month):02d}-{int(day):02d}"

    cn_match = re.match(r"^\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*$", value)
    if cn_match:
        year, month, day = cn_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return value


def standardize_doc_id(doc_id: Any) -> Optional[str]:
    if doc_id is None:
        return None
    if isinstance(doc_id, int):
        return f"doc{doc_id:03d}"

    value = str(doc_id).strip()
    if not value:
        return None
    match = re.fullmatch(r"doc0*(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"doc{int(match.group(1)):03d}"
    if value.isdigit():
        return f"doc{int(value):03d}"
    return value


def generate_doc_id(relative_path: str) -> str:
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def get_category_info(pdf_path: Path, root_dir: Path) -> Dict[str, Optional[str]]:
    try:
        parts = list(pdf_path.relative_to(root_dir).parts[:-1])
    except ValueError:
        parts = list(pdf_path.parts[:-1])

    category_path = "/".join(parts) if parts else None
    return {
        "category_path": category_path,
        "category_level_1": parts[0] if len(parts) > 0 else None,
        "category_level_2": parts[1] if len(parts) > 1 else None,
        "category_level_3": parts[2] if len(parts) > 2 else None,
        "category_level_4": parts[3] if len(parts) > 3 else None,
    }


def find_matching_json(pdf_path: Path) -> Optional[Path]:
    exact_json = pdf_path.with_suffix(".json")
    if exact_json.exists():
        return exact_json

    pdf_stem = pdf_path.stem
    simplified_stem = re.sub(r"[（(]\d+.*?[)）]", "", pdf_stem)
    simplified_stem = re.sub(r"\d+$", "", simplified_stem).strip()

    candidates: List[Tuple[float, Path]] = []
    for json_file in pdf_path.parent.glob("*.json"):
        score = SequenceMatcher(None, pdf_stem, json_file.stem).ratio()
        simple_score = SequenceMatcher(None, simplified_stem, json_file.stem).ratio()
        candidates.append((max(score, simple_score), json_file))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if candidates[0][0] >= 0.80:
        return candidates[0][1]
    return None


def load_manifest(pdf_path: Path, root_dir: Path) -> Tuple[Dict[str, Any], Optional[Path], Optional[str]]:
    json_path = find_matching_json(pdf_path)
    if not json_path:
        return {}, None, None
    try:
        return json.loads(json_path.read_text(encoding="utf-8")), json_path, None
    except Exception as exc:
        return {}, json_path, f"JSON 解析失败: {exc}"


def build_doc_info(pdf_path: Path, root_dir: Path) -> Tuple[Dict[str, Any], Optional[Path], Optional[str]]:
    relative_pdf = str(pdf_path.relative_to(root_dir))
    manifest, json_path, warning = load_manifest(pdf_path, root_dir)
    category_info = get_category_info(pdf_path, root_dir)
    doc_id = standardize_doc_id(manifest.get("doc_id")) or generate_doc_id(relative_pdf)

    doc_info = {
        "doc_id": doc_id,
        "doc_title": manifest.get("doc_title") or pdf_path.stem,
        "doc_no": manifest.get("doc_no"),
        "issuing_body": manifest.get("issuing_body"),
        "publish_date": normalize_date(manifest.get("publish_date")),
        "effective_date": normalize_date(manifest.get("effective_date")),
        "expiry_date": normalize_date(manifest.get("expiry_date")),
        "status": manifest.get("status") or "未知",
        "scope": manifest.get("scope"),
        "law_level": manifest.get("law_level"),
        "access_level": manifest.get("access_level") or "internal",
        "pdf_path": pdf_path,
        "relative_pdf": relative_pdf,
        **category_info,
    }
    if doc_info["status"] not in VALID_STATUSES:
        doc_info["status"] = "未知"
    return doc_info, json_path, warning


def normalize_law_markers(text: str) -> str:
    def replace(match: re.Match) -> str:
        number = re.sub(r"\s+", "", match.group(1))
        return f"第{number}{match.group(2)}"

    return LAW_MARKER_SPACE_RE.sub(replace, text)


def normalize_ocr_line(line: str) -> str:
    line = str(line).replace("\u3000", " ")
    line = re.sub(r"\s+", " ", line)
    return normalize_law_markers(line.strip())


def detect_toc_pages(lines: List[Dict[str, Any]]) -> Set[int]:
    by_page: Dict[int, List[str]] = {}
    for item in lines:
        page = item.get("page")
        text = normalize_ocr_line(item.get("text", ""))
        if isinstance(page, int) and text:
            by_page.setdefault(page, []).append(text)

    toc_pages: Set[int] = set()
    for page, texts in by_page.items():
        has_toc_title = any(text == "目录" for text in texts)
        structure_count = sum(
            1 for text in texts if CHAPTER_RE.match(text) or ARTICLE_RE.match(text)
        )
        if has_toc_title and structure_count >= 3:
            toc_pages.add(page)
    return toc_pages


def clean_ocr_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    toc_pages = detect_toc_pages(lines)
    cleaned: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, str]] = set()
    footer_keywords = ("内部资料", "仅供内部使用", "保密文件")

    for item in lines:
        page = item.get("page")
        if not isinstance(page, int) or page in toc_pages:
            continue
        text = normalize_ocr_line(item.get("text", ""))
        if not text or PAGE_NUM_RE.match(text):
            continue
        if any(keyword in text for keyword in footer_keywords):
            continue
        key = (page, text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"page": page, "text": text})
    return cleaned


def _collect_strings_from_old_paddle_shape(result: Any) -> List[str]:
    lines: List[str] = []
    if isinstance(result, (list, tuple)):
        if len(result) >= 2 and isinstance(result[1], (list, tuple)):
            candidate = result[1]
            if candidate and isinstance(candidate[0], str):
                lines.append(candidate[0])
                return lines
        for item in result:
            lines.extend(_collect_strings_from_old_paddle_shape(item))
    return lines


def extract_text_from_paddle_result(result: Any) -> List[str]:
    """
    从 PaddleOCR 返回结果中提取文本行，兼容 v2 的嵌套 tuple/list 和 v3 的 dict/OCRResult。
    """
    lines: List[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = normalize_ocr_line(value)
            if text:
                lines.append(text)
            return
        if isinstance(value, dict):
            for key in ("rec_texts", "texts", "text"):
                if key in value:
                    visit(value[key])
                    return
            for key in ("res", "result", "ocr_result", "data"):
                if key in value:
                    visit(value[key])
                    return
            return
        if hasattr(value, "json"):
            try:
                visit(value.json)
                return
            except Exception:
                pass
        if hasattr(value, "to_dict"):
            try:
                visit(value.to_dict())
                return
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            old_lines = _collect_strings_from_old_paddle_shape(value)
            if old_lines:
                for old_line in old_lines:
                    visit(old_line)
                return
            for item in value:
                visit(item)

    visit(result)

    deduped: List[str] = []
    seen: Set[str] = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    return deduped


def import_paddle_ocr_class():
    try:
        from paddleocr import PaddleOCR

        return PaddleOCR
    except Exception as exc:
        raise PaddleDependencyError(
            "未安装 paddleocr，请先安装 PaddleOCR 及其运行依赖。"
        ) from exc


def create_paddle_ocr():
    PaddleOCR = import_paddle_ocr_class()
    stable_cpu_engine_config = {
        "enable_new_ir": False,
        "delete_pass": ["onednn_placement_pass"],
    }
    attempts = [
        {
            "lang": "ch",
            "ocr_version": "PP-OCRv4",
            "use_textline_orientation": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "device": "cpu",
            "engine_config": stable_cpu_engine_config,
        },
        {"lang": "ch", "use_textline_orientation": True},
        {"lang": "ch", "use_angle_cls": True},
        {"lang": "ch"},
    ]
    errors: List[str] = []
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            errors.append(f"{kwargs}: {exc}")
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
    raise PaddleDependencyError(
        "未安装 paddleocr，请先安装 PaddleOCR 及其运行依赖。"
        + (" 初始化失败: " + " | ".join(errors[:2]) if errors else "")
    )


def run_paddle_ocr(ocr: Any, image_path: Path) -> List[str]:
    if hasattr(ocr, "predict"):
        result = ocr.predict(str(image_path))
    else:
        result = ocr.ocr(str(image_path), cls=True)
    return extract_text_from_paddle_result(result)


def render_pdf_page_to_image(page: Any, output_path: Path, zoom: float = 2.5) -> Path:
    import fitz

    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(str(output_path))
    return output_path


def _cache_is_compatible(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("page"), int)
        and isinstance(item.get("text"), str)
        for item in value
    )


def read_ocr_cache(cache_file: Path) -> Optional[List[Dict[str, Any]]]:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if _cache_is_compatible(data):
        return data
    return None


def write_ocr_cache(cache_file: Path, lines: List[Dict[str, Any]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")


def ocr_pdf_file(
    pdf_path: Path,
    doc_id: str,
    cache_dir: Path,
    ocr: Any,
    force_ocr: bool = False,
) -> List[Dict[str, Any]]:
    import fitz

    cache_file = cache_dir / f"{doc_id}_ocr_lines.json"
    if not force_ocr:
        cached = read_ocr_cache(cache_file)
        if cached is not None:
            return cached

    if ocr is None:
        ocr = create_paddle_ocr()

    lines: List[Dict[str, Any]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page_index in range(len(doc)):
                page_no = page_index + 1
                image_path = tmp_dir / f"page_{page_no:04d}.png"
                render_pdf_page_to_image(doc[page_index], image_path)
                page_lines = run_paddle_ocr(ocr, image_path)
                for text in page_lines:
                    clean_text = normalize_ocr_line(text)
                    if clean_text:
                        lines.append({"page": page_no, "text": clean_text})
                try:
                    image_path.unlink()
                except FileNotFoundError:
                    pass
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    write_ocr_cache(cache_file, lines)
    return lines


def parse_article(text: str) -> Optional[str]:
    match = ARTICLE_RE.match(text)
    return match.group(1) if match else None


def _flush_article(chunks: List[Dict[str, Any]], state: ArticleState, doc_info: Dict[str, Any]) -> None:
    if not state.article_no or not state.lines:
        return
    chunks.append(
        create_ocr_chunk(
            doc_info=doc_info,
            chapter=state.chapter,
            section=state.section,
            article_no=state.article_no,
            content="\n".join(state.lines).strip(),
            page_start=state.page_start or 1,
            page_end=state.page_end or state.page_start or 1,
            article_index=len(chunks) + 1,
        )
    )


def parse_structured_heading(text: str) -> Optional[Tuple[str, int]]:
    appendix_match = APPENDIX_HEADING_RE.match(text)
    if appendix_match and len(text) >= 2:
        return appendix_match.group(1).replace(" ", ""), 1

    chinese_match = CHINESE_HEADING_RE.match(text)
    if chinese_match:
        return chinese_match.group(1), 1

    dotted_match = NUMERIC_DOTTED_HEADING_RE.match(text)
    if dotted_match:
        marker = dotted_match.group(1)
        rest = dotted_match.group(2).strip()
        if rest or marker.count(".") >= 1:
            return marker, marker.count(".") + 1

    numeric_match = NUMERIC_TEXT_HEADING_RE.match(text)
    if numeric_match:
        marker = numeric_match.group(1)
        rest = numeric_match.group(2).strip()
        if rest and not re.match(r"^[.．、]?\s*[年月日发布实施]+$", rest):
            return marker, 1

    return None


def split_structured_fallback_chunks(
    lines: List[Dict[str, Any]], doc_info: Dict[str, Any]
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_marker: Optional[str] = None
    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None
    current_chapter: Optional[str] = None

    def flush() -> None:
        nonlocal current_lines, current_marker, current_page_start, current_page_end
        if not current_marker or not current_lines:
            return
        chunks.append(
            create_ocr_chunk(
                doc_info=doc_info,
                chapter=current_chapter,
                section=None,
                article_no=current_marker,
                content="\n".join(current_lines).strip(),
                page_start=current_page_start or 1,
                page_end=current_page_end or current_page_start or 1,
                article_index=len(chunks) + 1,
            )
        )

    for item in lines:
        page = item.get("page")
        line = normalize_ocr_line(item.get("text", ""))
        if not isinstance(page, int) or not line:
            continue
        heading = parse_structured_heading(line)
        if heading:
            marker, level = heading
            flush()
            current_marker = marker
            current_page_start = page
            current_page_end = page
            current_lines = [line]
            if level == 1:
                current_chapter = line
            continue
        if current_marker:
            current_lines.append(line)
            current_page_end = page

    flush()
    return chunks


def split_ocr_chunks(lines: List[Dict[str, Any]], doc_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    state = ArticleState()

    for item in lines:
        page = item.get("page")
        line = normalize_ocr_line(item.get("text", ""))
        if not isinstance(page, int) or not line:
            continue

        if CHAPTER_RE.match(line):
            state.chapter = line
            continue
        if SECTION_RE.match(line):
            state.section = line
            continue

        article_no = parse_article(line)
        if article_no:
            _flush_article(chunks, state, doc_info)
            state.article_no = article_no
            state.page_start = page
            state.page_end = page
            state.lines = [line]
            continue

        if state.article_no:
            state.lines.append(line)
            state.page_end = page

    _flush_article(chunks, state, doc_info)
    if chunks:
        return chunks
    return split_structured_fallback_chunks(lines, doc_info)


def create_ocr_chunk(
    doc_info: Dict[str, Any],
    chapter: Optional[str],
    section: Optional[str],
    article_no: str,
    content: str,
    page_start: int,
    page_end: int,
    article_index: int,
) -> Dict[str, Any]:
    doc_id = doc_info["doc_id"]
    return {
        "chunk_id": f"{doc_id}_article_{article_index:03d}",
        "doc_id": doc_id,
        "doc_title": doc_info.get("doc_title") or "",
        "doc_no": doc_info.get("doc_no"),
        "issuing_body": doc_info.get("issuing_body"),
        "publish_date": doc_info.get("publish_date"),
        "effective_date": doc_info.get("effective_date"),
        "expiry_date": doc_info.get("expiry_date"),
        "status": doc_info.get("status") or "未知",
        "scope": doc_info.get("scope"),
        "law_level": doc_info.get("law_level"),
        "category_path": doc_info.get("category_path"),
        "category_level_1": doc_info.get("category_level_1"),
        "category_level_2": doc_info.get("category_level_2"),
        "category_level_3": doc_info.get("category_level_3"),
        "category_level_4": doc_info.get("category_level_4"),
        "chapter": chapter,
        "section": section,
        "article_no": article_no,
        "clause_no": None,
        "content": content,
        "source_file": str(doc_info["pdf_path"]),
        "relative_source_file": str(doc_info["relative_pdf"]),
        "page_start": page_start,
        "page_end": page_end,
        "access_level": doc_info.get("access_level") or "internal",
        "parse_method": "ocr",
    }


def chunk_text_by_clauses(chunk: Dict[str, Any], seen_ids: Set[str]) -> List[Dict[str, Any]]:
    content = chunk.get("content") or ""
    if len(content) <= 1800:
        seen_ids.add(chunk.get("chunk_id"))
        return [chunk]

    matches = list(CLAUSE_SPLIT_RE.finditer(content))
    if len(matches) < 2:
        seen_ids.add(chunk.get("chunk_id"))
        return [chunk]

    prefix = content[: matches[0].start()].strip()
    article_no = chunk.get("article_no") or ""
    if article_no and prefix.startswith(article_no):
        prefix = prefix[len(article_no) :].strip()
    context_parts = [
        chunk.get("doc_title"),
        chunk.get("chapter"),
        article_no,
    ]
    context = " ".join(part for part in context_parts if part)
    if prefix:
        context = f"{context}\n{prefix}" if context else prefix

    base_id = chunk["chunk_id"]
    new_chunks: List[Dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        clause_content = content[start:end].strip()
        new_chunk = dict(chunk)
        new_chunk["clause_no"] = match.group(0)
        new_chunk["content"] = f"{context}\n{clause_content}".strip()
        candidate = f"{base_id}_clause_{index + 1:03d}"
        suffix = index + 1
        while candidate in seen_ids:
            suffix += 1
            candidate = f"{base_id}_clause_{suffix:03d}"
        new_chunk["chunk_id"] = candidate
        seen_ids.add(candidate)
        new_chunks.append(new_chunk)
    return new_chunks


def validate_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen: Set[str] = set()
    for index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            errors.append(f"Chunk {index}: chunk_id 为空")
        elif chunk_id in seen:
            errors.append(f"Chunk {index}: chunk_id '{chunk_id}' 重复")
        else:
            seen.add(chunk_id)
        if not chunk.get("doc_id"):
            errors.append(f"Chunk {index}: doc_id 为空")
        if not chunk.get("doc_title"):
            errors.append(f"Chunk {index}: doc_title 为空")
        if not chunk.get("content"):
            errors.append(f"Chunk {index}: content 为空")
        if not chunk.get("article_no"):
            errors.append(f"Chunk {index}: article_no 为空")
        if chunk.get("page_start") is None:
            errors.append(f"Chunk {index}: page_start 为空")
        if "category_path" not in chunk:
            errors.append(f"Chunk {index}: category_path 字段缺失")
        if not chunk.get("source_file"):
            errors.append(f"Chunk {index}: source_file 为空")
        if chunk.get("status") not in VALID_STATUSES:
            errors.append(f"Chunk {index}: status '{chunk.get('status')}' 不在有效值范围内")
        if chunk.get("parse_method") not in VALID_PARSE_METHODS:
            errors.append(
                f"Chunk {index}: parse_method '{chunk.get('parse_method')}' 不在有效值范围内"
            )
    return errors


def validate_ocr_quality(file_chunks: List[Dict[str, Any]], ocr_lines: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    total_chars = sum(len(item.get("text", "")) for item in ocr_lines)
    if total_chars < 100:
        errors.append(f"OCR 后文本总字符数少于 100（实际: {total_chars}）")
    if len(file_chunks) < 1:
        if not any(ARTICLE_RE.match(item.get("text", "")) for item in ocr_lines):
            errors.append("没有识别到任何“第X条”或可用结构标题")
        errors.append("chunk 数少于 1")
    for index, chunk in enumerate(file_chunks):
        if chunk.get("page_start") is None or chunk.get("page_end") is None:
            errors.append(f"Chunk {index}: page_start/page_end 缺失")
    return errors


def load_existing_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    if not ORIGINAL_CHUNKS.exists():
        return chunks
    with ORIGINAL_CHUNKS.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                chunk.setdefault("parse_method", "text")
                chunks.append(chunk)
    return chunks


def merge_chunks(
    original_chunks: List[Dict[str, Any]], ocr_chunks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    merged = []
    existing_ids: Set[str] = set()
    warnings: List[str] = []

    for chunk in original_chunks:
        original = dict(chunk)
        original.setdefault("parse_method", "text")
        merged.append(original)
        if original.get("chunk_id"):
            existing_ids.add(original["chunk_id"])

    collision_counts: Dict[str, int] = {}
    for chunk in ocr_chunks:
        new_chunk = dict(chunk)
        chunk_id = new_chunk.get("chunk_id")
        if chunk_id in existing_ids:
            collision_counts[chunk_id] = collision_counts.get(chunk_id, 0) + 1
            new_id = f"{chunk_id}_ocrfix_{collision_counts[chunk_id]:03d}"
            while new_id in existing_ids:
                collision_counts[chunk_id] += 1
                new_id = f"{chunk_id}_ocrfix_{collision_counts[chunk_id]:03d}"
            warnings.append(f"OCR chunk_id 冲突: {chunk_id} -> {new_id}")
            new_chunk["chunk_id"] = new_id
            new_chunk["_collision_resolved"] = True
        merged.append(new_chunk)
        if new_chunk.get("chunk_id"):
            existing_ids.add(new_chunk["chunk_id"])
    return merged, warnings


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_ocr_review(doc_info: Dict[str, Any], chunks: List[Dict[str, Any]], ocr_page_count: int) -> None:
    if not chunks:
        return
    OCR_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OCR_REVIEW_DIR / f"{doc_info['doc_id']}.txt"
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(f"文件：{doc_info['relative_pdf']}\n")
        handle.write(f"分类：{doc_info.get('category_path')}\n")
        handle.write(f"识别页数：{ocr_page_count}\n")
        handle.write(f"生成 chunk 数：{len(chunks)}\n")
        for index, chunk in enumerate(chunks, start=1):
            handle.write(f"==================== chunk {index} ====================\n")
            handle.write(f"chunk_id: {chunk.get('chunk_id')}\n")
            handle.write(f"chapter: {chunk.get('chapter')}\n")
            handle.write(f"article_no: {chunk.get('article_no')}\n")
            handle.write(f"page: {chunk.get('page_start')}-{chunk.get('page_end')}\n")
            handle.write("content:\n")
            handle.write((chunk.get("content") or "") + "\n")


def make_file_report(
    pdf_path: Path,
    root_dir: Path,
    doc_info: Optional[Dict[str, Any]] = None,
    json_path: Optional[Path] = None,
    status: str = "failed",
    error: Optional[str] = None,
    warning: Optional[str] = None,
    ocr_lines: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    relative_pdf = str(pdf_path.relative_to(root_dir))
    category_info = get_category_info(pdf_path, root_dir)
    ocr_lines = ocr_lines or []
    chunks = chunks or []
    pages = {item["page"] for item in ocr_lines if isinstance(item.get("page"), int)}
    return {
        "pdf": str(pdf_path.absolute()),
        "relative_pdf": relative_pdf,
        "json": str(json_path.absolute()) if json_path else None,
        "relative_json": str(json_path.relative_to(root_dir)) if json_path else None,
        "category_path": (doc_info or category_info).get("category_path"),
        "status": status,
        "ocr_page_count": len(pages),
        "ocr_text_char_count": sum(len(item.get("text", "")) for item in ocr_lines),
        "chunk_count": len(chunks),
        "error": error,
        "warning": warning,
    }


def process_failed_pdf(
    pdf_path: Path,
    root_dir: Path,
    ocr: Any,
    force_ocr: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    doc_info, json_path, warning = build_doc_info(pdf_path, root_dir)
    try:
        raw_lines = ocr_pdf_file(pdf_path, doc_info["doc_id"], OCR_CACHE_DIR, ocr, force_ocr)
        cleaned_lines = clean_ocr_lines(raw_lines)
        chunks: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for chunk in split_ocr_chunks(cleaned_lines, doc_info):
            chunks.extend(chunk_text_by_clauses(chunk, seen_ids))

        quality_errors = validate_ocr_quality(chunks, cleaned_lines)
        if quality_errors:
            return (
                make_file_report(
                    pdf_path,
                    root_dir,
                    doc_info,
                    json_path,
                    error="OCR 质量检查失败: " + "; ".join(quality_errors[:3]),
                    warning=warning,
                    ocr_lines=cleaned_lines,
                ),
                [],
            )

        validation_errors = validate_chunks(chunks)
        if validation_errors:
            return (
                make_file_report(
                    pdf_path,
                    root_dir,
                    doc_info,
                    json_path,
                    error="校验失败: " + "; ".join(validation_errors[:3]),
                    warning=warning,
                    ocr_lines=cleaned_lines,
                ),
                [],
            )

        save_ocr_review(doc_info, chunks, len({line["page"] for line in cleaned_lines}))
        return (
            make_file_report(
                pdf_path,
                root_dir,
                doc_info,
                json_path,
                status="success",
                warning=warning,
                ocr_lines=cleaned_lines,
                chunks=chunks,
            ),
            chunks,
        )
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        if isinstance(exc, PaddleDependencyError):
            detail = str(exc)
        return (
            make_file_report(
                pdf_path,
                root_dir,
                doc_info,
                json_path,
                error=f"OCR 处理失败: {detail}",
                warning=warning,
            ),
            [],
        )


def load_parse_report() -> Dict[str, Any]:
    if not INPUT_PARSE_REPORT.exists():
        raise FileNotFoundError(f"parse_report.json 不存在: {INPUT_PARSE_REPORT}")
    data = json.loads(INPUT_PARSE_REPORT.read_text(encoding="utf-8"))
    if not isinstance(data.get("file_reports"), list):
        raise ValueError("parse_report.json 中没有可用的 file_reports")
    return data


def find_target_failed_pdfs(parse_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    targets = []
    for report in parse_report.get("file_reports", []):
        error = report.get("error") or ""
        if report.get("status") == "failed" and any(keyword in error for keyword in OCR_ERROR_KEYWORDS):
            targets.append(report)
    return targets


def write_report(report: Dict[str, Any]) -> None:
    OUTPUT_OCR_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_OCR_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_final_report(
    target_count: int,
    file_reports: List[Dict[str, Any]],
    ocr_chunks: List[Dict[str, Any]],
    merged_chunks: List[Dict[str, Any]],
    validation_errors: List[str],
) -> Dict[str, Any]:
    return {
        "root_dir": str(ROOT_DIR),
        "input_parse_report": str(INPUT_PARSE_REPORT),
        "original_chunks": str(ORIGINAL_CHUNKS),
        "ocr_chunks": str(OUTPUT_OCR_CHUNKS),
        "merged_chunks": str(OUTPUT_MERGED_CHUNKS),
        "target_failed_pdf_count": target_count,
        "ocr_success_files": sum(1 for report in file_reports if report.get("status") == "success"),
        "ocr_failed_files": sum(1 for report in file_reports if report.get("status") != "success"),
        "ocr_chunk_count": len(ocr_chunks),
        "merged_chunk_count": len(merged_chunks),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "file_reports": file_reports,
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("\nOCR 补救处理完成")
    print(f"根目录：{ROOT_DIR}")
    print(f"待 OCR 失败 PDF 数：{report['target_failed_pdf_count']}")
    print(f"OCR 成功文件数：{report['ocr_success_files']}")
    print(f"OCR 失败文件数：{report['ocr_failed_files']}")
    print(f"OCR 新增 chunk 数：{report['ocr_chunk_count']}")
    print(f"合并后 chunk 数：{report['merged_chunk_count']}")
    print(f"OCR chunk 输出：{OUTPUT_OCR_CHUNKS}")
    print(f"合并输出：{OUTPUT_MERGED_CHUNKS}")
    print(f"报告：{OUTPUT_OCR_REPORT}")
    print(f"人工核对目录：{OCR_REVIEW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描版 PDF OCR 补救脚本")
    parser.add_argument("--force-ocr", action="store_true", help="忽略缓存，重新 OCR")
    args = parser.parse_args()

    OUTPUT_OCR_CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OCR_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        parse_report = load_parse_report()
        targets = find_target_failed_pdfs(parse_report)
    except Exception as exc:
        print(f"错误：{exc}")
        return

    print(f"开始处理根目录：{ROOT_DIR}")
    print(f"待 OCR 失败 PDF 数：{len(targets)}")

    ocr_chunks: List[Dict[str, Any]] = []
    file_reports: List[Dict[str, Any]] = []

    ocr = None

    for target in tqdm(targets, desc="处理 OCR"):
        pdf_path = Path(target.get("pdf") or ROOT_DIR / target.get("relative_pdf", ""))
        if not pdf_path.exists():
            file_reports.append(
                make_file_report(pdf_path, ROOT_DIR, error=f"PDF 不存在: {pdf_path}")
            )
            continue

        file_report, chunks = process_failed_pdf(pdf_path, ROOT_DIR, ocr, args.force_ocr)
        file_reports.append(file_report)
        ocr_chunks.extend(chunks)

    save_jsonl(OUTPUT_OCR_CHUNKS, ocr_chunks)

    original_chunks = load_existing_chunks()
    merged_chunks, collision_warnings = merge_chunks(original_chunks, ocr_chunks)
    save_jsonl(OUTPUT_MERGED_CHUNKS, merged_chunks)

    validation_errors = validate_chunks(merged_chunks)
    validation_errors.extend(collision_warnings)

    report = build_final_report(
        target_count=len(targets),
        file_reports=file_reports,
        ocr_chunks=ocr_chunks,
        merged_chunks=merged_chunks,
        validation_errors=validation_errors,
    )
    write_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
