import pdfplumber

from src.extractors.base import BaseExtractor
from src.models import (
    BoundingBox,
    BlockType,
    ExtractedBlock,
    ExtractedDocument,
)
from src.utils import load_rules


def _page_confidence(
    char_count: int,
    char_density: float,
    image_ratio: float,
    font_ratio: float,
    char_count_min: int,
    image_ratio_max: float,
    font_meta_weight: float,
    default_conf: float,
) -> float:
    if char_count >= char_count_min and image_ratio <= image_ratio_max:
        base = min(1.0, 0.5 + 0.5 * (char_count / max(char_count_min * 2, 1)))
        font_bonus = font_ratio * font_meta_weight
        return min(1.0, default_conf + font_bonus)
    if char_count < char_count_min or image_ratio > image_ratio_max:
        return max(0.0, default_conf - 0.3 - (1.0 - font_ratio) * 0.2)
    return default_conf


class FastTextExtractor(BaseExtractor):
    DEFAULT_CONFIDENCE = 0.75

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("strategies", {}).get("fast", {})
        self.char_count_min = cfg.get("char_count_min", 50)
        self.image_ratio_max = cfg.get("image_ratio_max", 0.6)
        self.font_meta_weight = cfg.get("font_meta_weight", 0.3)
        self.default_conf = float(cfg.get("confidence", self.DEFAULT_CONFIDENCE))

    def extract(self, source) -> ExtractedDocument:
        rules = load_rules()
        cfg = rules.get("strategies", {}).get("fast", {})
        char_count_min = cfg.get("char_count_min", self.char_count_min)
        image_ratio_max = cfg.get("image_ratio_max", self.image_ratio_max)
        font_meta_weight = cfg.get("font_meta_weight", self.font_meta_weight)
        default_conf = float(cfg.get("confidence", self.default_conf))

        blocks: list[ExtractedBlock] = []
        page_confidences: list[float] = []
        with pdfplumber.open(source) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                (x0, top, x1, bottom) = (
                    page.bbox if hasattr(page, "bbox") else (0, 0, page.width, page.height)
                )
                area = (x1 - x0) * (bottom - top) if (x1 > x0 and bottom > top) else 1.0
                chars = page.chars or []
                char_count = len(chars)
                chars_with_font = sum(1 for c in chars if c.get("fontname"))
                font_ratio = chars_with_font / char_count if char_count else 0.0
                image_area = 0.0
                if page.images:
                    for im in page.images:
                        image_area += (im.get("width") or 0) * (im.get("height") or 0)
                image_ratio = image_area / area if area else 0.0
                char_density = char_count / area if area else 0.0

                pc = _page_confidence(
                    char_count, char_density, image_ratio, font_ratio,
                    char_count_min, image_ratio_max, font_meta_weight, default_conf,
                )
                page_confidences.append(pc)

                bbox = BoundingBox(x0=x0, y0=top, x1=x1, y1=bottom, page=i + 1)
                blocks.append(
                    ExtractedBlock(text=text.strip(), bbox=bbox, block_type=BlockType.TEXT)
                )

        confidence = min(page_confidences) if page_confidences else default_conf
        reading_order = [str(k) for k in range(len(blocks))]
        return ExtractedDocument(
            blocks=blocks,
            reading_order=reading_order,
            confidence=confidence,
            strategy_used="fast_text",
        )
