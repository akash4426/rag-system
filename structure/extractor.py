"""Structure extraction for PDF, DOCX, Markdown, and plain text.

Strategy order for PDFs: bookmarks/ToC, then heading detection by font size and
boldness, then one root section with fixed-size chunks. This is a pragmatic
rule-based extractor; it is not a reproduction of any published system.
"""

from __future__ import annotations

import hashlib
import logging
import re
import statistics
from pathlib import Path

from structure.models import ExtractedDocument, Section

logger = logging.getLogger(__name__)

_HEADING_STYLE = re.compile(r"^heading\s*([1-6])$", re.IGNORECASE)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TEXT_HEADING = re.compile(r"^(chapter|section|part)\s+\d+(?:\.\d+)*[.:\s]+(.+)$", re.IGNORECASE)
_SLIDE_ONLY = re.compile(r"^slide\s+\d+$", re.IGNORECASE)


def make_doc_id(path: str | Path) -> str:
    """Content-stable document ID that is safe for section IDs."""
    path = Path(path)
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40] or "document"
    return f"{slug}-{digest}"


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip())


def _finalize(sections: list[Section], doc_id: str, strategy: str) -> list[Section]:
    """Assign IDs, ordinals, and parent links from a level stack."""
    stack: list[Section] = []
    finalized: list[Section] = []
    for ordinal, raw in enumerate(sections):
        if raw.level < 1:
            raw.level = 1
        while stack and stack[-1].level >= raw.level:
            stack.pop()
        parent_id = stack[-1].section_id if stack else None
        raw.section_id = f"{doc_id}::s{ordinal:04d}"
        raw.parent_id = parent_id
        raw.ordinal = ordinal
        raw.doc_id = doc_id
        raw.strategy = strategy
        raw.page_start = max(1, int(raw.page_start or 1))
        raw.page_end = max(raw.page_start, int(raw.page_end or raw.page_start))
        stack.append(raw)
        finalized.append(raw)
    return finalized


def _pdf_toc_sections(pdf, doc_id: str) -> list[Section]:
    """Build sections from PDF bookmarks, with page ranges from hierarchy."""
    entries = []
    for item in pdf.get_toc(simple=True):
        if len(item) < 3:
            continue
        level, title, page = item[0], _clean_title(item[1]), int(item[2])
        if page < 1 or page > pdf.page_count or not title:
            continue
        entries.append({"level": max(1, int(level)), "title": title, "page": page})

    sections: list[Section] = []
    for index, entry in enumerate(entries):
        end_page = pdf.page_count
        for later in entries[index + 1:]:
            if later["level"] <= entry["level"]:
                end_page = max(entry["page"], later["page"] - 1)
                break
        text = "\n".join(
            pdf[page - 1].get_text("text")
            for page in range(entry["page"], end_page + 1)
            if 0 <= page - 1 < pdf.page_count
        )
        sections.append(
            Section(
                section_id="",
                title=entry["title"],
                level=entry["level"],
                parent_id=None,
                page_start=entry["page"],
                page_end=end_page,
                text=text.strip(),
            )
        )
    return sections


def _pdf_heading_candidates(pdf) -> list[dict]:
    """Detect heading-like lines from font size, boldness, and repetition."""
    lines: list[dict] = []
    size_weights: list[float] = []
    page_frequency: dict[str, int] = {}

    for page_number in range(1, pdf.page_count + 1):
        page = pdf[page_number - 1]
        data = page.get_text("dict")
        seen_on_page: set[str] = set()
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = _clean_title("".join(span.get("text", "") for span in spans))
                if not text:
                    continue
                sizes = [float(span.get("size", 0.0)) for span in spans if span.get("text", "").strip()]
                size = max(sizes) if sizes else 0.0
                bold = any(int(span.get("flags", 0)) & 16 for span in spans)
                bbox = line.get("bbox", (0, 0, 0, 0))
                lines.append(
                    {
                        "page": page_number,
                        "y": float(bbox[1]),
                        "text": text,
                        "size": size,
                        "bold": bold,
                    }
                )
                for span in spans:
                    span_text = span.get("text", "").strip()
                    if span_text and size >= 4:
                        size_weights.extend([float(span.get("size", 0.0))] * min(len(span_text), 80))
                normalized = text.lower()
                if normalized not in seen_on_page:
                    page_frequency[normalized] = page_frequency.get(normalized, 0) + 1
                    seen_on_page.add(normalized)

    if not size_weights:
        return []
    body_size = statistics.median(size_weights)
    repeated_threshold = max(2, int(pdf.page_count * 0.4))

    candidates = []
    seen: set[tuple[int, float, str]] = set()
    for line in lines:
        text = line["text"]
        if len(text) > 160 or len(text) < 2:
            continue
        if re.fullmatch(r"[\d\s./-]+", text):
            continue
        if pdf.page_count >= 4 and page_frequency.get(text.lower(), 0) > repeated_threshold:
            continue
        is_large = line["size"] >= body_size * 1.12
        is_bold_heading = line["bold"] and line["size"] >= body_size * 1.02 and len(text) <= 100
        if not (is_large or is_bold_heading):
            continue
        key = (line["page"], line["y"], text)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({**line, "level_hint": 1 if is_large else 2})

    # Preserve reading order and cap pathological font-size noise.
    candidates.sort(key=lambda item: (item["page"], item["y"]))
    return candidates[:200]


def _pdf_text_slice(pdf, start_page: int, start_y: float, end_page: int, end_y: float) -> str:
    """Extract text blocks between two heading positions in reading order."""
    parts: list[str] = []
    for page_number in range(start_page, end_page + 1):
        page = pdf[page_number - 1]
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
            position = (page_number, float(y0))
            if position < (start_page, start_y):
                continue
            if (page_number, float(y0)) >= (end_page, end_y):
                continue
            clean = text.strip()
            if clean:
                parts.append(clean)
    return "\n".join(parts)


def _pdf_heading_sections(pdf, doc_id: str) -> list[Section]:
    """Build sections from detected headings and their block ranges."""
    headings = _pdf_heading_candidates(pdf)
    if not headings:
        return []

    # Map distinct heading sizes to levels, with bold-only headings one deeper.
    sizes = sorted({round(item["size"], 1) for item in headings}, reverse=True)
    for heading in headings:
        size_level = sizes.index(round(heading["size"], 1)) + 1
        heading["level"] = min(4, max(size_level, heading["level_hint"] if heading["bold"] else size_level))

    sections: list[Section] = []
    for index, heading in enumerate(headings):
        if index + 1 < len(headings):
            nxt = headings[index + 1]
            end_page, end_y = nxt["page"], nxt["y"]
        else:
            end_page, end_y = pdf.page_count + 1, 0.0
        text = _pdf_text_slice(
            pdf, heading["page"], heading["y"], min(end_page, pdf.page_count + 1), end_y
        )
        if heading["text"] not in text:
            text = f"{heading['text']}\n{text}".strip()
        sections.append(
            Section(
                section_id="",
                title=heading["text"],
                level=heading["level"],
                parent_id=None,
                page_start=heading["page"],
                page_end=min(max(heading["page"], end_page), pdf.page_count),
                text=text,
            )
        )
    return sections


def _whole_document_section(text: str, title: str, page_count: int, level: int = 1) -> list[Section]:
    return [
        Section(
            section_id="",
            title=title,
            level=level,
            parent_id=None,
            page_start=1,
            page_end=max(1, page_count),
            text=text.strip(),
        )
    ]


def _extract_pdf(path: Path, doc_id: str) -> ExtractedDocument:
    import pymupdf

    with pymupdf.open(path) as pdf:
        page_count = pdf.page_count
        pdf_title = _clean_title((pdf.metadata or {}).get("title", "")) or path.stem
        toc = pdf.get_toc(simple=True)

        if len(toc) >= 2:
            sections = _pdf_toc_sections(pdf, doc_id)
            strategy = "pdf_bookmarks"
        else:
            sections = _pdf_heading_sections(pdf, doc_id)
            strategy = "pdf_headings" if sections else "fixed_chunks"

        if not sections:
            text = "\n".join(page.get_text("text") for page in pdf)
            sections = _whole_document_section(text, pdf_title, page_count)
            strategy = "fixed_chunks"

        sections = _finalize(sections, doc_id, strategy)
        return ExtractedDocument(
            doc_id=doc_id,
            filename=path.name,
            path=str(path),
            title=pdf_title,
            strategy=strategy,
            page_count=page_count,
            sections=sections,
        )


def _extract_docx(path: Path, doc_id: str) -> ExtractedDocument:
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    sections: list[Section] = []
    current: Section | None = None
    title = path.stem
    saw_heading = False

    def append_body(text: str) -> None:
        nonlocal current
        if not text.strip():
            return
        if current is None:
            current = Section(
                section_id="", title=title, level=1, parent_id=None,
                page_start=1, page_end=1, text="",
            )
            sections.append(current)
        current.text = f"{current.text}\n{text}".strip() if current.text else text.strip()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = (paragraph.style.name or "").strip()
        match = _HEADING_STYLE.match(style_name)
        is_title = style_name.lower() == "title"
        if match or is_title:
            saw_heading = True
            level = 1 if is_title else int(match.group(1))
            title = text if is_title and title == path.stem else title
            current = Section(
                section_id="", title=text, level=level, parent_id=None,
                page_start=1, page_end=1, text="",
            )
            sections.append(current)
        else:
            append_body(text)

    # Include table rows in the current section; DOCX has no reliable pages.
    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        append_body("\n".join(row for row in rows if row.strip()))

    if not saw_heading:
        text = "\n".join(section.text for section in sections).strip()
        sections = _whole_document_section(text, title, 1)

    strategy = "docx_headings" if saw_heading else "fixed_chunks"
    sections = _finalize(sections, doc_id, strategy)
    return ExtractedDocument(
        doc_id=doc_id,
        filename=path.name,
        path=str(path),
        title=title,
        strategy=strategy,
        page_count=1,
        sections=sections,
    )


def _extract_text(path: Path, doc_id: str) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    is_markdown = path.suffix.lower() in {".md", ".markdown"}
    sections: list[Section] = []
    current: Section | None = None
    preamble: list[str] = []
    in_code_fence = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            if current is not None:
                current.text = f"{current.text}\n{line}".strip()
            else:
                preamble.append(line)
            continue

        match = None
        if not in_code_fence:
            if is_markdown:
                match = _MD_HEADING.match(line)
                level = len(match.group(1)) if match else 0
                heading_text = match.group(2) if match else ""
            else:
                match = _TEXT_HEADING.match(line)
                level = 1 if match else 0
                heading_text = match.group(2) if match else ""
        else:
            level, heading_text = 0, ""

        if match:
            current = Section(
                section_id="", title=_clean_title(heading_text), level=level,
                parent_id=None, page_start=1, page_end=1, text="",
            )
            sections.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current.text = f"{current.text}\n{line}".strip() if current.text else line

    if preamble:
        preamble_text = "\n".join(preamble).strip()
        if sections:
            sections[0].text = f"{preamble_text}\n\n{sections[0].text}".strip()
        else:
            sections = _whole_document_section(preamble_text, path.stem, 1)

    if not sections:
        sections = _whole_document_section(text, path.stem, 1)
        strategy = "fixed_chunks"
    elif is_markdown:
        strategy = "markdown_headings"
    else:
        strategy = "text_headings"

    sections = _finalize(sections, doc_id, strategy)
    return ExtractedDocument(
        doc_id=doc_id,
        filename=path.name,
        path=str(path),
        title=sections[0].title if sections else path.stem,
        strategy=strategy,
        page_count=1,
        sections=sections,
    )


def extract_document(path: str | Path) -> ExtractedDocument:
    """Extract a document's section tree and record which strategy was used."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    doc_id = make_doc_id(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        result = _extract_pdf(path, doc_id)
    elif suffix == ".docx":
        result = _extract_docx(path, doc_id)
    elif suffix in {".txt", ".md", ".markdown"}:
        result = _extract_text(path, doc_id)
    else:
        raise ValueError(f"Unsupported structure format: {suffix}")

    result.sections = [section for section in result.sections if section.title or section.text]
    logger.info(
        "Structure extraction %s: strategy=%s sections=%d pages=%d",
        path.name, result.strategy, len(result.sections), result.page_count,
    )
    print(
        f"  ✓ Structure [{path.name}]: {result.strategy}, "
        f"{len(result.sections)} sections"
    )
    return result


def extract_structure(path: str | Path) -> list[Section]:
    """Return the extracted section list for a document."""
    return extract_document(path).sections
