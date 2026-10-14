#!/usr/bin/env python3
"""
递归扫描法规语料库目录，生成条文级 chunks.jsonl

该脚本用于：
- 递归扫描指定目录下的所有 PDF 文件
- 兼容对应的 JSON 元数据文件
- 抽取 PDF 文本并进行清洗
- 识别法规结构（章、节、条等）
- 按法规结构切分 chunk
- 输出 chunks.jsonl 和 parse_report.json
"""

import json
import os
import re
import hashlib
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Set
from tqdm import tqdm

# ==================== 配置参数 ====================
ROOT_DIR = Path(os.environ.get("CDC_LAW_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUTPUT_CHUNKS = ROOT_DIR / "_rag_output" / "chunks.jsonl"
OUTPUT_REPORT = ROOT_DIR / "_rag_output" / "parse_report.json"

# 要忽略的目录
IGNORE_DIRS = {"_rag_output", "scripts", ".git", "__pycache__", ".venv", "venv"}

# ==================== 正则表达式 ====================
CN_NUM = r"[一二三四五六七八九十百千万零〇0-9]+"
CHAPTER_RE = re.compile(rf"^第{CN_NUM}章\s*(.*)")
SECTION_RE = re.compile(rf"^第{CN_NUM}节\s*(.*)")

# 条文匹配 - 支持"第一条"、"第1条"等格式（使用中文数字）
ARTICLE_RE = re.compile(r"^第([一二三四五六七八九十百千万零〇0-9]+)条\s*(.*)")

# 款的匹配 - （一）、（二）等（添加第一个捕获组）
CLAUSE_RE = re.compile(r"^（([一二三四五六七八九十百千万零〇0-9]+)）(.*)")

# 项的匹配 - 1.、1、等格式
ITEM_RE = re.compile(r"^[0-9]+[\.．、]")

# 目录页检测
TOC_KEYWORDS_RE = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+章|第[一二三四五六七八九十百千万零〇0-9]+条|目录")
PAGE_NUM_RE = re.compile(r"^\s*[-]?\d+[-–—]?\s*$")

# ==================== 辅助函数 ====================

def normalize_date(value: Optional[str]) -> str:
    """
    标准化日期格式为 YYYY-MM-DD
    
    支持的格式：
    - 2019.3.2
    - 2019-3-2
    - 2019/3/2
    - 2019_3_2
    - 2019年3月2日
    
    如果无法识别，保留原字符串
    """
    if not value or not isinstance(value, str):
        return None
    
    value = value.strip()
    
    # 尝试各种分隔符的组合（修复正则表达式）
    import re
    patterns = [
        r"^(\d{4})[.年](\d{1,2})[.月/\\_\-]?\s*(\d{1,2})日?",  # 2019.3.2 或 2019年3月2日
    ]
    
    for pattern in patterns:
        try:
            match = re.match(pattern, value)
            if match:
                year, month, day = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"
        except Exception:
            continue
    
    # 尝试直接的分隔符格式
    separators = ['-', '/', '_', '.']
    for sep in separators:
        if sep in value:
            parts = value.split(sep)
            if len(parts) >= 3:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    return f"{year}-{month:02d}-{day:02d}"
                except ValueError:
                    continue
    
    # 返回原字符串
    return value


def generate_doc_id(relative_path: str) -> str:
    """
    根据相对路径生成稳定的 doc_id
    
    使用 MD5 哈希的前 12 位
    """
    hash_value = hashlib.md5(relative_path.encode('utf-8')).hexdigest()[:12]
    return f"doc_{hash_value}"


def standardize_doc_id(doc_id: Any) -> str:
    """
    标准化 doc_id 格式
    
    - 如果是数字 75，转换为 "doc075"
    - 如果是 "doc75"，转换为 "doc075"
    - 如果是 "doc075"，保持不变
    - 其他情况生成新的 ID
    """
    if doc_id is None:
        return None
    
    if isinstance(doc_id, int):
        return f"doc{doc_id:03d}"
    
    doc_id = str(doc_id).strip()
    
    # 如果已经是 "docXXX" 格式，保持不变
    if re.match(r'^doc\d+$', doc_id):
        return doc_id
    
    # 如果是纯数字
    if doc_id.isdigit():
        return f"doc{int(doc_id):03d}"
    
    # 其他情况生成新的 ID
    return None


def get_category_info(pdf_path: Path, root_dir: Path) -> Dict[str, Any]:
    """
    从 PDF 路径中提取分类信息
    
    返回：
    - category_path: 相对于 root_dir 的路径（不包含文件名）
    - category_level_1 到 category_level_4: 各级目录
    """
    try:
        rel_path = pdf_path.relative_to(root_dir)
        parts = list(rel_path.parts[:-1])  # 去掉文件名
        
        if not parts:
            return {
                "category_path": None,
                "category_level_1": None,
                "category_level_2": None,
                "category_level_3": None,
                "category_level_4": None
            }
        
        category_path = "/".join(parts)
        
        result = {
            "category_path": category_path,
            "category_level_1": parts[0] if len(parts) >= 1 else None,
            "category_level_2": parts[1] if len(parts) >= 2 else None,
            "category_level_3": parts[2] if len(parts) >= 3 else None,
            "category_level_4": parts[3] if len(parts) >= 4 else None
        }
        
        return result
        
    except ValueError:
        # 无法计算相对路径
        return {
            "category_path": None,
            "category_level_1": None,
            "category_level_2": None,
            "category_level_3": None,
            "category_level_4": None
        }


def find_matching_json(pdf_path: Path) -> Optional[Path]:
    """
    查找与 PDF 文件匹配的 JSON 文件
    
    匹配规则：
    1. 同目录、同 basename 的 JSON
    2. 同目录下 basename 高度相似的 JSON
    """
    json_dir = pdf_path.parent
    pdf_stem = pdf_path.stem
    
    # 规则 1: 完全匹配
    exact_json = json_dir / f"{pdf_stem}.json"
    if exact_json.exists():
        return exact_json
    
    # 规则 2: 高度相似的匹配（移除版本号等后缀）
    # 尝试移除括号内的内容和数字后缀
    import re
    simplified_name = re.sub(r'[（(]\d+.*?[)）]', '', pdf_stem)
    simplified_name = re.sub(r'[0-9]+$', '', simplified_name).strip()
    
    for json_file in json_dir.glob("*.json"):
        if json_file.stem == f"{simplified_name}.json":
            continue
        # 计算相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, pdf_stem, json_file.stem).ratio()
        if similarity > 0.8:
            return json_file
    
    # 尝试其他常见变体
    variants = [
        f"{simplified_name}.json",
        f"{pdf_stem.replace('（', '(').replace('）', ')')}.json",
    ]
    
    for variant in variants:
        json_path = json_dir / variant
        if json_path.exists():
            return json_path
    
    return None


def extract_text_from_pdf(pdf_path: Path, skip_pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    从 PDF 中抽取文本
    
    返回：
    - list of dict with keys: "page_num", "text"
    """
    try:
        import fitz
        
        doc = fitz.open(str(pdf_path))
        
        if skip_pages is None:
            skip_pages = []
        
        pages_text = []
        for page_num in range(len(doc)):
            # 页码从 1 开始
            actual_page_num = page_num + 1
            
            if actual_page_num in skip_pages:
                continue
            
            page = doc[page_num]
            text = page.get_text("text", sort=True)
            
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    lines.append(line)
            
            if lines:  # 只有非空页才添加
                pages_text.append({
                    "page_num": actual_page_num,
                    "lines": lines,
                    "text": '\n'.join(lines)
                })
        
        doc.close()
        return pages_text
        
    except Exception as e:
        raise Exception(f"PDF 无法打开: {str(e)}")


def normalize_line(line: str) -> str:
    """
    规范化单行文本
    
    - 全角空格替换为普通空格
    - 多个空格合并为一个
    """
    # 全角空格替换为普通空格
    line = line.replace('\u3000', ' ')
    # 多个空格合并为一个
    line = re.sub(r'\s+', ' ', line)
    return line.strip()


def clean_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清洗文本行列表
    
    - 删除单独页码（如 "1-"、"第 1 页"）
    - 删除明显页眉页脚
    """
    cleaned = []
    
    for item in lines:
        page_num = item["page_num"]
        clean_page_lines = []
        
        for line in item.get("lines", []):
            line = normalize_line(line)
            
            # 跳过单独的页码
            if PAGE_NUM_RE.match(line):
                continue
            
            # 跳过明显页眉页脚
            footer_keywords = ["内部资料", "仅供内部使用", "保密文件"]
            if any(keyword in line for keyword in footer_keywords):
                continue
            
            # 跳过空行
            if not line:
                continue
            
            clean_page_lines.append(line)
        
        if clean_page_lines:
            cleaned.append({
                "page_num": page_num,
                "lines": clean_page_lines
            })
    
    return cleaned


def detect_toc_pages(pages: List[Dict[str, Any]]) -> Set[int]:
    """
    检测目录页
    
    保守策略：只有在页面非常明显是目录时才跳过
    """
    toc_pages = set()
    
    for page in pages:
        lines = page.get("lines", [])
        
        # 统计章、节、条的出现次数
        toc_count = 0
        total_lines = len(lines)
        
        if total_lines < 3:  # 太短的页不考虑
            continue
        
        for line in lines:
            if TOC_KEYWORDS_RE.search(line):
                toc_count += 1
        
        # 如果超过一半的行包含目录关键词，认为是目录页
        if total_lines > 0 and (toc_count / total_lines) > 0.5:
            toc_pages.add(page["page_num"])
    
    return toc_pages


# ==================== 法规结构识别 ====================

def parse_chapter(title: str) -> Optional[Dict[str, Any]]:
    """解析章标题"""
    match = CHAPTER_RE.match(title)
    if match:
        return {
            "type": "chapter",
            "full_text": match.group(0),
            "title": match.group(1) or ""
        }
    return None


def parse_section(title: str) -> Optional[Dict[str, Any]]:
    """解析节标题"""
    match = SECTION_RE.match(title)
    if match:
        return {
            "type": "section",
            "full_text": match.group(0),
            "title": match.group(1) or ""
        }
    return None


def parse_article(text: str) -> Optional[Dict[str, Any]]:
    """解析条文"""
    # 尝试匹配"第X条"
    match = ARTICLE_RE.match(text)
    if match:
        num = match.group(1)
        content = match.group(2) or ""
        
        return {
            "type": "article",
            "article_no": f"第{num}条",
            "content": content
        }
    
    # 尝试匹配阿拉伯数字开头的条文
    ar_num_re = re.compile(r"^第(\d+)条(.*)")
    match = ar_num_re.match(text)
    if match:
        return {
            "type": "article",
            "article_no": f"第{match.group(1)}条",
            "content": match.group(2) or ""
        }
    
    return None


def parse_clause(text: str) -> Optional[Dict[str, Any]]:
    """解析款"""
    match = CLAUSE_RE.match(text)
    if match:
        return {
            "type": "clause",
            "clause_no": match.group(1),
            "content": match.group(2) or ""
        }
    return None


# ==================== Chunk 切分 ====================

@dataclass
class ArticleContext:
    """文章上下文信息"""
    chapter: Optional[str] = None
    section: Optional[str] = None
    page_start: int = 0
    page_end: int = 0


def split_into_chunks(pages_text: List[Dict], doc_info: Dict) -> List[Dict]:
    """
    按法规结构切分 chunk
    
    以"条"为单位进行切分
    """
    chunks = []
    
    # 当前上下文
    current_context = ArticleContext()
    current_content_lines = []
    article_no = None
    
    all_lines = []
    for page in pages_text:
        all_lines.append({
            "page_num": page["page_num"],
            "lines": page.get("lines", [])
        })
    
    # 检测目录页
    toc_pages = detect_toc_pages(pages_text)
    
    # 遍历所有行
    for page in pages_text:
        if page["page_num"] in toc_pages:
            continue
        
        page_num = page["page_num"]
        
        for line in page.get("lines", []):
            line = normalize_line(line)
            
            # 跳过目录页上的行
            if any(TOC_KEYWORDS_RE.search(l) for l in [line]):
                # 但要检查是否只是包含"目录"字样的普通段落
                pass
            
            # 检查是否是新的条文开始
            article_match = parse_article(line)
            
            if article_match:
                # 结束上一条，保存 chunk（如果有内容）
                if current_content_lines and article_no:
                    content = '\n'.join(current_content_lines)
                    
                    chunk = create_chunk(
                        doc_info=doc_info,
                        chapter=current_context.chapter,
                        section=current_context.section,
                        article_no=article_no,
                        content=content,
                        page_start=current_context.page_start,
                        page_end=current_context.page_end,
                        chunks_count=len(chunks) + 1
                    )
                    chunks.append(chunk)
                
                # 开始新条文
                current_content_lines = [line]
                article_no = article_match["article_no"]
                current_context.page_start = page_num
                
                # 更新上下文
                if not current_context.chapter:
                    current_context.chapter = None
                if not current_context.section:
                    current_context.section = None
                    
            else:
                # 继续当前条文内容
                # 检查是否是款
                clause_match = parse_clause(line)
                
                if clause_match and current_content_lines:
                    # 这是款的开始，可能需要拆分（后续实现）
                    current_content_lines.append(line)
                elif not clause_match:
                    # 普通行内容
                    current_content_lines.append(line)
    
    # 保存最后一条
    if current_content_lines and article_no:
        content = '\n'.join(current_content_lines)
        
        chunk = create_chunk(
            doc_info=doc_info,
            chapter=current_context.chapter,
            section=current_context.section,
            article_no=article_no,
            content=content,
            page_start=current_context.page_start,
            page_end=all_lines[-1]["page_num"] if all_lines else 0,
            chunks_count=len(chunks) + 1
        )
        chunks.append(chunk)
    
    return chunks


def create_chunk(doc_info: Dict, chapter: Optional[str], section: Optional[str],
                article_no: str, content: str, page_start: int, page_end: int,
                chunks_count: int) -> Dict:
    """创建一个 chunk"""
    # 生成 chunk_id
    doc_id = doc_info["doc_id"]
    chunk_id = f"{doc_id}_article_{chunks_count:03d}"
    
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_title": doc_info.get("doc_title", ""),
        "doc_no": doc_info.get("doc_no"),
        "issuing_body": doc_info.get("issuing_body"),
        "publish_date": doc_info.get("publish_date"),
        "effective_date": doc_info.get("effective_date"),
        "expiry_date": doc_info.get("expiry_date"),
        "status": doc_info.get("status", "未知"),
        "scope": doc_info.get("scope"),
        "law_level": doc_info.get("law_level"),
        "category_path": doc_info["category_path"],
        "category_level_1": doc_info["category_level_1"],
        "category_level_2": doc_info["category_level_2"],
        "category_level_3": doc_info["category_level_3"],
        "category_level_4": doc_info["category_level_4"],
        "chapter": chapter,
        "section": section,
        "article_no": article_no,
        "clause_no": None,
        "content": content,
        "source_file": str(doc_info["pdf_path"]),
        "relative_source_file": str(doc_info["relative_pdf"]),
        "page_start": page_start,
        "page_end": page_end,
        "access_level": doc_info.get("access_level", "internal")
    }


def chunk_text_by_clauses(chunk: Dict, seen_ids: Set[str]) -> List[Dict]:
    """
    对超长条文按款进行二次切分
    
    如果 content 超过 1800 个中文字符，则按款（（一）、（二）等）切分
    使用已使用ID集合确保 chunk_id 唯一
    """
    MAX_LENGTH = 1800
    content = chunk.get("content", "")
    
    # 检查是否需要切分
    if len(content) <= MAX_LENGTH:
        return [chunk]
    
    # 尝试按款切分
    clause_pattern = r'（[一二三四五六七八九十百千万零〇0-9]+）'
    parts = re.split(f'({clause_pattern})', content)
    
    if len(parts) < 3:  # 没有找到款，无法切分
        return [chunk]
    
    # 合并内容
    new_chunks = []
    current_content = ""
    clause_no = None
    
    for i in range(0, len(parts), 2):
        part = parts[i]
        
        if i + 1 < len(parts):
            clause_part = parts[i + 1]
            
            # 新的一款
            if current_content:
                new_chunk = chunk.copy()
                new_chunk["content"] = current_content.strip()
                new_chunk["clause_no"] = clause_no
                
                # 修改 chunk_id - 确保唯一性
                old_id = new_chunk["chunk_id"]
                article_part = old_id.rsplit("_article_", 1)[0]
                
                # 尝试生成唯一ID，直到找到未使用的
                base_clause_num = len(new_chunks) + 1
                clause_num = base_clause_num
                while True:
                    new_chunk_id = f"{article_part}_clause_{clause_num:03d}"
                    if new_chunk_id not in seen_ids:
                        break
                    clause_num += 1
                
                new_chunk["chunk_id"] = new_chunk_id
                seen_ids.add(new_chunk_id)
                
                new_chunks.append(new_chunk)
            
            clause_no = clause_part
            current_content = f"{part}{clause_part}"
        else:
            # 最后一部分
            if part.strip():
                current_content += part
    
    # 添加最后一部分
    if current_content.strip():
        new_chunk = chunk.copy()
        new_chunk["content"] = current_content.strip()
        new_chunk["clause_no"] = clause_no
        
        old_id = new_chunk["chunk_id"]
        article_part = old_id.rsplit("_article_", 1)[0]
        
        # 尝试生成唯一ID，直到找到未使用的
        base_clause_num = len(new_chunks) + 1
        clause_num = base_clause_num
        while True:
            new_chunk_id = f"{article_part}_clause_{clause_num:03d}"
            if new_chunk_id not in seen_ids:
                break
            clause_num += 1
        
        new_chunk["chunk_id"] = new_chunk_id
        seen_ids.add(new_chunk_id)
        
        new_chunks.append(new_chunk)
    
    return new_chunks if new_chunks else [chunk]


# ==================== 质量校验 ====================

def validate_chunks(chunks: List[Dict]) -> List[str]:
    """
    校验 chunks 是否符合要求
    
    返回错误列表
    """
    errors = []
    chunk_ids = set()
    
    valid_statuses = {"现行有效", "已废止", "部分有效", "未知"}
    
    for i, chunk in enumerate(chunks):
        # 检查必填字段
        if not chunk.get("chunk_id"):
            errors.append(f"Chunk {i}: chunk_id 为空")
        
        if chunk.get("chunk_id") in chunk_ids:
            errors.append(f"Chunk {i}: chunk_id '{chunk['chunk_id']}' 重复")
        else:
            chunk_ids.add(chunk["chunk_id"])
        
        if not chunk.get("doc_id"):
            errors.append(f"Chunk {i}: doc_id 为空")
        
        if not chunk.get("doc_title"):
            errors.append(f"Chunk {i}: doc_title 为空")
        
        if not chunk.get("content"):
            errors.append(f"Chunk {i}: content 为空")
        
        if not chunk.get("article_no"):
            errors.append(f"Chunk {i}: article_no 为空")
        
        if chunk.get("page_start") is None:
            errors.append(f"Chunk {i}: page_start 为空")
        
        # 检查分类字段
        if "category_path" not in chunk:
            errors.append(f"Chunk {i}: category_path 字段缺失")
        
        # 检查 source_file
        if not chunk.get("source_file"):
            errors.append(f"Chunk {i}: source_file 为空")
        
        # 检查 status
        status = chunk.get("status")
        if status and status not in valid_statuses:
            errors.append(f"Chunk {i}: status '{status}' 不在有效值范围内")
    
    return errors


# ==================== 主处理流程 ====================

def process_single_pdf(pdf_path: Path, root_dir: Path) -> Dict[str, Any]:
    """
    处理单个 PDF 文件
    
    返回：file_report dict 和 chunks list
    """
    file_report = {
        "pdf": str(pdf_path.absolute()),
        "relative_pdf": str(pdf_path.relative_to(root_dir)),
        "json": None,
        "relative_json": None,
        "category_path": None,
        "status": "failed",
        "chunk_count": 0,
        "error": None
    }
    
    try:
        # 获取分类信息
        category_info = get_category_info(pdf_path, root_dir)
        
        # 加载 JSON（如果存在）
        json_path = find_matching_json(pdf_path)
        manifest = {}
        
        if json_path and json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                
                file_report["json"] = str(json_path.absolute())
                file_report["relative_json"] = str(json_path.relative_to(root_dir))
            except Exception as e:
                file_report["warning"] = f"JSON 解析失败: {str(e)}"
        
        # 提取基本元数据
        doc_info = {
            "doc_id": manifest.get("doc_id"),
            "doc_title": manifest.get("doc_title") or pdf_path.stem,
            "doc_no": manifest.get("doc_no"),
            "issuing_body": manifest.get("issuing_body"),
            "publish_date": normalize_date(manifest.get("publish_date")),
            "effective_date": normalize_date(manifest.get("effective_date")),
            "expiry_date": normalize_date(manifest.get("expiry_date")),
            "status": manifest.get("status", "未知") or "未知",
            "scope": manifest.get("scope"),
            "law_level": manifest.get("law_level"),
            "access_level": manifest.get("access_level", "internal") or "internal",
            "category_path": category_info["category_path"],
            "category_level_1": category_info["category_level_1"],
            "category_level_2": category_info["category_level_2"],
            "category_level_3": category_info["category_level_3"],
            "category_level_4": category_info["category_level_4"],
            "pdf_path": pdf_path,
            "relative_pdf": pdf_path.relative_to(root_dir)
        }
        
        # 标准化 doc_id
        standardized_id = standardize_doc_id(doc_info["doc_id"])
        if standardized_id:
            doc_info["doc_id"] = standardized_id
        else:
            # 生成新的 doc_id
            doc_info["doc_id"] = generate_doc_id(str(pdf_path.relative_to(root_dir)))
        
        # 提取 skip_pages（如果 JSON 中有）
        skip_pages = manifest.get("skip_pages", [])
        if isinstance(skip_pages, int):
            skip_pages = [skip_pages]
        
        # 抽取文本
        pages_text = extract_text_from_pdf(pdf_path, skip_pages)
        
        if not pages_text:
            file_report["error"] = "抽取不到文本，可能是扫描版 PDF，需要 OCR"
            return file_report, []
        
        # 清洗文本
        clean_pages = clean_lines(pages_text)
        
        # 切分 chunk
        chunks = split_into_chunks(clean_pages, doc_info)
        
        # 对超长条文进行二次切分
        seen_ids: Set[str] = set()  # 跟踪已使用的ID
        all_chunks = []
        for chunk in chunks:
            sub_chunks = chunk_text_by_clauses(chunk, seen_ids)
            all_chunks.extend(sub_chunks)
        
        # 校验 chunks
        validation_errors = validate_chunks(all_chunks)
        
        if validation_errors:
            file_report["error"] = "校验失败: " + "; ".join(validation_errors[:3])
            return file_report, []
        
        file_report["status"] = "success"
        file_report["chunk_count"] = len(all_chunks)
        file_report["category_path"] = category_info["category_path"]
        
        return file_report, all_chunks
        
    except Exception as e:
        file_report["error"] = f"处理失败: {str(e)}"
        traceback_str = traceback.format_exc()
        if "PyMuPDF" in traceback_str or "fitz" in traceback_str:
            file_report["error"] = "抽取不到文本，可能是扫描版 PDF，需要 OCR"
        return file_report, []


def main():
    """主函数"""
    print(f"开始处理根目录: {ROOT_DIR}")
    
    # 收集所有 PDF 文件
    pdf_files = []
    for pdf_path in ROOT_DIR.rglob("*.pdf"):
        # 检查是否在忽略的目录中
        rel_parts = list(pdf_path.relative_to(ROOT_DIR).parts[:-1])
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
        
        pdf_files.append(pdf_path)
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    
    # 处理每个 PDF
    all_chunks = []
    file_reports = []
    
    for pdf_path in tqdm(pdf_files, desc="处理 PDF"):
        file_report, chunks = process_single_pdf(pdf_path, ROOT_DIR)
        
        if chunks:
            all_chunks.extend(chunks)
        
        file_reports.append(file_report)
    
    # 生成统计信息
    success_count = sum(1 for r in file_reports if r["status"] == "success")
    failed_count = len(file_reports) - success_count
    
    # 写入 chunks.jsonl
    with open(OUTPUT_CHUNKS, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"写入 {len(all_chunks)} 个 chunks 到 {OUTPUT_CHUNKS}")
    
    # 构建最终报告
    report = {
        "root_dir": str(ROOT_DIR),
        "output_chunks": str(OUTPUT_CHUNKS),
        "total_pdf_files": len(pdf_files),
        "success_files": success_count,
        "failed_files": failed_count,
        "total_chunks": len(all_chunks),
        "validation_error_count": 0,
        "validation_errors": [],
        "file_reports": file_reports
    }
    
    # 写入 parse_report.json
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"写入报告到 {OUTPUT_REPORT}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    print(f"根目录: {ROOT_DIR}")
    print(f"PDF 总数: {len(pdf_files)}")
    print(f"成功文件数: {success_count}")
    print(f"失败文件数: {failed_count}")
    print(f"生成 chunk 数: {len(all_chunks)}")
    print(f"输出: {OUTPUT_CHUNKS}")
    print(f"报告: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
