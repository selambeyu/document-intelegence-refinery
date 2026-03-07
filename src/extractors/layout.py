from pathlib import Path

from src.extractors.base import BaseExtractor
from src.models import BoundingBox, BlockType, ExtractedBlock, ExtractedDocument, ExtractedTable

try:
    from docling.document_converter import DocumentConverter
    from src.adapters.docling_adapter import docling_result_to_document
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False

import pdfplumber


def _extract_pdfplumber(source) -> ExtractedDocument:
    path = Path(source) if isinstance(source, (Path, str)) else None
    if path is not None and not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    blocks: list[ExtractedBlock] = []
    tables: list[ExtractedTable] = []
    with pdfplumber.open(source) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            (x0, top, x1, bottom) = (
                page.bbox if hasattr(page, "bbox") else (0, 0, page.width, page.height)
            )
            bbox = BoundingBox(x0=x0, y0=top, x1=x1, y1=bottom, page=i + 1)
            blocks.append(
                ExtractedBlock(text=text.strip(), bbox=bbox, block_type=BlockType.TEXT)
            )
            for t in page.find_tables() or []:
                data = t.extract()
                if data:
                    headers = data[0] if data else []
                    rows = data[1:] if len(data) > 1 else []
                    tables.append(
                        ExtractedTable(
                            headers=[str(h) for h in headers],
                            rows=[[str(c) for c in row] for row in rows],
                            bbox=bbox,
                            page=i + 1,
                        )
                    )
    reading_order = [str(k) for k in range(len(blocks))]
    return ExtractedDocument(
        blocks=blocks,
        reading_order=reading_order,
        tables=tables,
        confidence=0.90,
        strategy_used="layout",
    )


class LayoutExtractor(BaseExtractor):
    DEFAULT_CONFIDENCE = 0.90

    def __init__(self, prefer_docling: bool = True):
        self.prefer_docling = prefer_docling

    def extract(self, source) -> ExtractedDocument:
        path = Path(source) if isinstance(source, (Path, str)) else None
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        if self.prefer_docling and _DOCLING_AVAILABLE and path is not None:
            converter = DocumentConverter()
            result = converter.convert(str(path))
            return docling_result_to_document(result)
        return _extract_pdfplumber(source)
