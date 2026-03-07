import hashlib
import re
from typing import Any

from src.models import (
    BoundingBox,
    BlockType,
    ChunkType,
    ExtractedBlock,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    LDU,
)

_CROSS_REF_PATTERNS = [
    (re.compile(r"\b(?:see\s+)?(?:Table\s+)(\d+)\b", re.I), "table"),
    (re.compile(r"\b(?:see\s+)?(?:Figure\s+)(\d+)\b", re.I), "figure"),
    (re.compile(r"\b(?:Appendix\s+)([A-Z0-9]+)\b", re.I), "appendix"),
    (re.compile(r"\b(?:Section\s+)(\d+(?:\.\d+)*)\b", re.I), "section"),
]


def _token_count(text: str) -> int:
    return max(0, (len(text.split()) * 4) // 3)


def _content_hash(content: str, page_refs: list[int], bbox: BoundingBox | None) -> str:
    parts = [content, repr(sorted(page_refs))]
    if bbox:
        parts.append(f"{bbox.page}:{bbox.x0},{bbox.y0},{bbox.x1},{bbox.y1}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _is_heading(text: str, max_length: int = 80) -> bool:
    s = text.strip()
    if not s or len(s) > max_length:
        return False
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    if re.match(r"^(?:Chapter|Section|Part|\d+\.?)\s", lines[0], re.I):
        return True
    return len(lines[0]) <= 60 and not lines[0].endswith(".")


def _is_numbered_list(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return bool(re.match(r"^\d+[.)]\s", lines[0])) and bool(re.match(r"^\d+[.)]\s", lines[1]))


def _resolve_cross_refs(ldus: list[LDU]) -> None:
    label_to_hash: dict[str, str] = {}
    table_idx = 0
    figure_idx = 0
    for ldu in ldus:
        if ldu.chunk_type == ChunkType.TABLE:
            table_idx += 1
            label_to_hash[f"Table {table_idx}"] = ldu.content_hash or ""
        elif ldu.chunk_type == ChunkType.FIGURE:
            figure_idx += 1
            label_to_hash[f"Figure {figure_idx}"] = ldu.content_hash or ""

    for ldu in ldus:
        if ldu.chunk_type not in (ChunkType.TEXT, ChunkType.LIST) or not ldu.content:
            continue
        refs: list[dict[str, str]] = []
        for pat, kind in _CROSS_REF_PATTERNS:
            for m in pat.finditer(ldu.content):
                label = m.group(0).strip()
                num = m.group(1) if m.lastindex else ""
                key = f"Table {num}" if kind == "table" else (f"Figure {num}" if kind == "figure" else label)
                target_hash = label_to_hash.get(key)
                if target_hash:
                    refs.append({"label": label, "target_content_hash": target_hash})
        if refs:
            meta = dict(ldu.metadata) if ldu.metadata else {}
            meta["cross_refs"] = refs
            ldu.metadata = meta


class ChunkValidator:
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def validate_single(self, ldu: LDU) -> list[str]:
        errors: list[str] = []
        if ldu.chunk_type == ChunkType.TABLE:
            if not ldu.content.strip():
                errors.append("TABLE has empty content (missing header/rows)")
            else:
                lines = ldu.content.strip().split("\n")
                if not lines:
                    errors.append("TABLE has no header row")
        if ldu.chunk_type == ChunkType.TEXT and ldu.token_count > self.max_tokens:
            errors.append(f"TEXT exceeds max_tokens ({ldu.token_count} > {self.max_tokens})")
        if ldu.content.strip() and not (ldu.content_hash or "").strip():
            errors.append("missing content_hash")
        return errors

    def validate(self, ldus: list[LDU]) -> list[str]:
        errors: list[str] = []
        for i, ldu in enumerate(ldus):
            for e in self.validate_single(ldu):
                errors.append(f"LDU[{i}] {e}")
        return errors

    def filter_valid(self, ldus: list[LDU]) -> list[LDU]:
        return [ldu for ldu in ldus if not self.validate_single(ldu)]


class ChunkingEngine:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.max_tokens = int(cfg.get("max_tokens", 512))
        self.overlap_tokens = int(cfg.get("overlap_tokens", 32))
        self.min_chunk_tokens = int(cfg.get("min_chunk_tokens", 16))
        self.respect_heading_boundaries = cfg.get("respect_heading_boundaries", True)
        self.respect_table_boundaries = cfg.get("respect_table_boundaries", True)

    def chunk(self, doc: ExtractedDocument) -> list[LDU]:
        ldus: list[LDU] = []
        current_section = ""
        text_parts: list[tuple[str, BoundingBox, int]] = []
        emitted_tables: set[int] = set()
        emitted_figures: set[int] = set()

        order = [int(x) for x in doc.reading_order if str(x).strip().isdigit()]
        if not order and doc.blocks:
            order = list(range(len(doc.blocks)))

        for idx in order:
            if idx < 0 or idx >= len(doc.blocks):
                continue
            block = doc.blocks[idx]
            if block.block_type == BlockType.TABLE or (block.table_ref is not None):
                self._flush_text_ldus(text_parts, current_section, ldus)
                text_parts = []
                ref = block.table_ref
                if ref is not None and ref not in emitted_tables and ref < len(doc.tables):
                    t = doc.tables[ref]
                    ldu = self._table_to_ldu(t, current_section)
                    ldus.append(ldu)
                    emitted_tables.add(ref)
                continue
            if block.block_type == BlockType.FIGURE or (block.figure_ref is not None):
                self._flush_text_ldus(text_parts, current_section, ldus)
                text_parts = []
                ref = block.figure_ref
                if ref is not None and ref not in emitted_figures and ref < len(doc.figures):
                    f = doc.figures[ref]
                    ldu = self._figure_to_ldu(f, current_section)
                    ldus.append(ldu)
                    emitted_figures.add(ref)
                continue
            text = (block.text or "").strip()
            if not text:
                continue
            if self.respect_heading_boundaries and _is_heading(text):
                self._flush_text_ldus(text_parts, current_section, ldus)
                text_parts = []
                current_section = text
                continue
            bbox = block.bbox
            page = bbox.page if bbox else 1
            if _is_numbered_list(text) and self.respect_table_boundaries:
                tc = _token_count(text)
                if tc <= self.max_tokens:
                    self._flush_text_ldus(text_parts, current_section, ldus)
                    text_parts = []
                    page_refs = [page]
                    ldus.append(LDU(
                        content=text,
                        chunk_type=ChunkType.LIST,
                        page_refs=page_refs,
                        bbox=bbox,
                        parent_section=current_section,
                        token_count=tc,
                        content_hash=_content_hash(text, page_refs, bbox),
                    ))
                    continue
            text_parts.append((text, bbox, page))

        self._flush_text_ldus(text_parts, current_section, ldus)

        for ref, t in enumerate(doc.tables):
            if ref in emitted_tables:
                continue
            ldus.append(self._table_to_ldu(t, ""))

        for ref, f in enumerate(doc.figures):
            if ref in emitted_figures:
                continue
            ldus.append(self._figure_to_ldu(f, ""))

        _resolve_cross_refs(ldus)
        return ldus

    def _table_to_ldu(self, t: ExtractedTable, parent_section: str) -> LDU:
        lines = ["\t".join(t.headers)]
        for row in t.rows:
            lines.append("\t".join(str(c) for c in row))
        content = "\n".join(lines)
        page_refs = [t.page] if t.page else ([t.bbox.page] if t.bbox else [0])
        bbox = t.bbox
        tc = _token_count(content)
        return LDU(
            content=content,
            chunk_type=ChunkType.TABLE,
            page_refs=page_refs,
            bbox=bbox,
            parent_section=parent_section,
            token_count=tc,
            content_hash=_content_hash(content, page_refs, bbox),
        )

    def _figure_to_ldu(self, f: ExtractedFigure, parent_section: str) -> LDU:
        content = f"Figure: {f.caption}" if f.caption else "Figure"
        page_refs = [f.page] if f.page else ([f.bbox.page] if f.bbox else [0])
        bbox = f.bbox
        tc = _token_count(content)
        meta = {"caption": f.caption} if f.caption else {}
        return LDU(
            content=content,
            chunk_type=ChunkType.FIGURE,
            page_refs=page_refs,
            bbox=bbox,
            parent_section=parent_section,
            token_count=tc,
            content_hash=_content_hash(content, page_refs, bbox),
            metadata=meta,
        )

    def _flush_text_ldus(
        self,
        text_parts: list[tuple[str, BoundingBox, int]],
        parent_section: str,
        ldus: list[LDU],
    ) -> None:
        if not text_parts:
            return
        combined = "\n\n".join(t[0] for t in text_parts)
        tokens = _token_count(combined)
        if tokens <= self.max_tokens:
            page_refs = sorted(set(t[2] for t in text_parts))
            bbox = text_parts[0][1] if text_parts else None
            content = combined
            if tokens >= self.min_chunk_tokens or not ldus:
                ldus.append(LDU(
                    content=content,
                    chunk_type=ChunkType.TEXT,
                    page_refs=page_refs,
                    bbox=bbox,
                    parent_section=parent_section,
                    token_count=tokens,
                    content_hash=_content_hash(content, page_refs, bbox),
                ))
            return
        current: list[str] = []
        current_tokens = 0
        current_pages: set[int] = set()
        current_bbox: BoundingBox | None = None
        for text, bbox, page in text_parts:
            for para in text.split("\n\n"):
                pt = _token_count(para)
                if current_tokens + pt > self.max_tokens and current:
                    content = "\n\n".join(current)
                    ldus.append(LDU(
                        content=content,
                        chunk_type=ChunkType.TEXT,
                        page_refs=sorted(current_pages),
                        bbox=current_bbox,
                        parent_section=parent_section,
                        token_count=current_tokens,
                        content_hash=_content_hash(content, sorted(current_pages), current_bbox),
                    ))
                    current = []
                    current_tokens = 0
                    current_pages = set()
                current.append(para)
                current_tokens += pt
                current_pages.add(page)
                if current_bbox is None:
                    current_bbox = bbox
        if current:
            content = "\n\n".join(current)
            ldus.append(LDU(
                content=content,
                chunk_type=ChunkType.TEXT,
                page_refs=sorted(current_pages),
                bbox=current_bbox,
                parent_section=parent_section,
                token_count=current_tokens,
                content_hash=_content_hash(content, sorted(current_pages), current_bbox),
            ))
