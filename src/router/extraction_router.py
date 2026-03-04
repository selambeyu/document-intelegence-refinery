from pathlib import Path
from typing import BinaryIO

from src.models import DocumentProfile, ExtractedDocument, ExtractionCost, OriginType
from src.extractors import FastTextExtractor, LayoutExtractor, VisionExtractor

try:
    import pytesseract
    _VISION_FALLBACK_ERRORS = (pytesseract.TesseractNotFoundError, ImportError)
except ImportError:
    _VISION_FALLBACK_ERRORS = (ImportError,)

CONFIDENCE_ESCALATION_THRESHOLD = 0.6


class ExtractionRouter:
    def __init__(
        self,
        fast_extractor: FastTextExtractor | None = None,
        layout_extractor: LayoutExtractor | None = None,
        vision_extractor: VisionExtractor | None = None,
        confidence_threshold: float = CONFIDENCE_ESCALATION_THRESHOLD,
    ):
        self._fast = fast_extractor or FastTextExtractor()
        self._layout = layout_extractor or LayoutExtractor()
        self._vision = vision_extractor or VisionExtractor()
        self._confidence_threshold = confidence_threshold

    def extract(self, source: Path | str | BinaryIO, profile: DocumentProfile) -> ExtractedDocument:
        doc = self._extract_with_strategy(source, profile)
        if doc.confidence < self._confidence_threshold:
            doc = self._layout.extract(source)
            doc = doc.model_copy(update={"strategy_used": doc.strategy_used + "_escalated"})
        return doc

    def _extract_with_strategy(
        self, source: Path | str | BinaryIO, profile: DocumentProfile
    ) -> ExtractedDocument:
        if profile.origin_type == OriginType.SCANNED_IMAGE:
            try:
                return self._vision.extract(source)
            except _VISION_FALLBACK_ERRORS:
                doc = self._layout.extract(source)
                return doc.model_copy(update={"strategy_used": "layout_vision_fallback"})
        if profile.extraction_cost == ExtractionCost.FAST_TEXT_SUFFICIENT:
            return self._fast.extract(source)
        if profile.extraction_cost == ExtractionCost.NEEDS_VISION_MODEL:
            try:
                return self._vision.extract(source)
            except _VISION_FALLBACK_ERRORS:
                doc = self._layout.extract(source)
                return doc.model_copy(update={"strategy_used": "layout_vision_fallback"})
        return self._layout.extract(source)
