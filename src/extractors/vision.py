import io
from pathlib import Path

from src.extractors.base import BaseExtractor
from src.models import BoundingBox, BlockType, ExtractedBlock, ExtractedDocument
from src.utils import load_rules

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    from src.extractors.vision_client import (
        extract_document_with_vision,
        get_vision_api_key,
    )
    _VISION_CLIENT_AVAILABLE = True
except ImportError:
    _VISION_CLIENT_AVAILABLE = False


def _ocr_extract(source) -> ExtractedDocument:
    if fitz is None or pytesseract is None or Image is None:
        raise ImportError("pymupdf, pytesseract and Pillow are required for OCR fallback")
    path = Path(source) if isinstance(source, (Path, str)) else None
    if path is not None and not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    doc = fitz.open(source)
    blocks: list[ExtractedBlock] = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        rect = page.rect
        bbox = BoundingBox(
            x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1, page=i + 1
        )
        blocks.append(
            ExtractedBlock(text=text.strip(), bbox=bbox, block_type=BlockType.TEXT)
        )
    doc.close()
    reading_order = [str(k) for k in range(len(blocks))]
    return ExtractedDocument(
        blocks=blocks,
        reading_order=reading_order,
        confidence=0.65,
        strategy_used="vision_ocr",
    )


class VisionExtractor(BaseExtractor):
    DEFAULT_CONFIDENCE = 0.65

    def __init__(self, prefer_vlm: bool = True):
        self.prefer_vlm = prefer_vlm

    def extract(self, source) -> ExtractedDocument:
        if fitz is None:
            raise ImportError("pymupdf (fitz) is required for VisionExtractor")
        path = Path(source) if isinstance(source, (Path, str)) else None
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        rules = load_rules()
        vision_cfg = rules.get("strategies", {}).get("vision", {})
        budget = vision_cfg.get("budget_per_document")
        cost_per_page = vision_cfg.get("cost_per_page", 0.02)

        if self.prefer_vlm and _VISION_CLIENT_AVAILABLE:
            api_key = get_vision_api_key()
            if api_key and path is not None:
                try:
                    doc = fitz.open(source)
                    page_images: list[bytes] = []
                    for i in range(len(doc)):
                        page = doc[i]
                        pix = page.get_pixmap(dpi=150)
                        page_images.append(pix.tobytes("png"))
                    doc.close()
                    raw_blocks, cost_actual = extract_document_with_vision(
                        page_images,
                        api_key,
                        cost_per_page=cost_per_page,
                        budget=budget,
                    )
                    blocks = []
                    for b in raw_blocks:
                        page = b.get("page", 1)
                        bbox = BoundingBox(
                            x0=float(b.get("x0", 0)),
                            y0=float(b.get("y0", 0)),
                            x1=float(b.get("x1", 0)),
                            y1=float(b.get("y1", 0)),
                            page=page,
                        )
                        blocks.append(
                            ExtractedBlock(
                                text=(b.get("text") or "").strip(),
                                bbox=bbox,
                                block_type=BlockType.TEXT,
                            )
                        )
                    if not blocks:
                        raise ValueError("VLM returned no blocks")
                    reading_order = [str(k) for k in range(len(blocks))]
                    return ExtractedDocument(
                        blocks=blocks,
                        reading_order=reading_order,
                        confidence=self.DEFAULT_CONFIDENCE,
                        strategy_used="vision",
                        cost_actual=cost_actual,
                    )
                except Exception:
                    pass
                doc_ocr = _ocr_extract(source)
                return doc_ocr.model_copy(
                    update={"strategy_used": "vision_ocr_fallback"}
                )

        if pytesseract is None or Image is None:
            raise ImportError("pytesseract and Pillow are required for VisionExtractor")
        return _ocr_extract(source)
