from pathlib import Path
from typing import BinaryIO

from src.models import DocumentProfile, ExtractedDocument, ExtractionCost, OriginType
from src.extractors import FastTextExtractor, LayoutExtractor, VisionExtractor
from src.utils import load_rules

try:
    import pytesseract
    _VISION_FALLBACK_ERRORS = (pytesseract.TesseractNotFoundError, ImportError)
except ImportError:
    _VISION_FALLBACK_ERRORS = (ImportError,)


def _get_router_config(config: dict | None = None) -> dict:
    rules = config or load_rules()
    return rules.get("router", {})


class ExtractionRouter:
    def __init__(
        self,
        fast_extractor: FastTextExtractor | None = None,
        layout_extractor: LayoutExtractor | None = None,
        vision_extractor: VisionExtractor | None = None,
        confidence_threshold: float | None = None,
        review_threshold: float | None = None,
        config: dict | None = None,
    ):
        rcfg = _get_router_config(config)
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else float(rcfg.get("confidence_escalation_threshold", 0.6))
        )
        self._review_threshold = (
            review_threshold
            if review_threshold is not None
            else float(rcfg.get("review_threshold", 0.5))
        )
        self._config = config
        self._fast = fast_extractor or FastTextExtractor(config=config)
        self._layout = layout_extractor or LayoutExtractor()
        self._vision = vision_extractor or VisionExtractor()

    def extract(self, source: Path | str | BinaryIO, profile: DocumentProfile) -> ExtractedDocument:
        doc = self._extract_with_escalation(source, profile)
        review_flag = doc.confidence < self._review_threshold
        return doc.model_copy(update={"review_flag": review_flag})

    def _extract_with_escalation(
        self, source: Path | str | BinaryIO, profile: DocumentProfile
    ) -> ExtractedDocument:
        doc = self._extract_with_strategy(source, profile)
        if doc.confidence >= self._confidence_threshold:
            return doc
        doc = self._escalate_to_layout(source, doc)
        if doc.confidence >= self._confidence_threshold:
            return doc
        doc = self._escalate_to_vision_if_appropriate(source, profile, doc)
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

    def _escalate_to_layout(self, source: Path | str | BinaryIO, doc: ExtractedDocument) -> ExtractedDocument:
        if "layout" in doc.strategy_used and "escalated" not in doc.strategy_used:
            return doc
        next_doc = self._layout.extract(source)
        return next_doc.model_copy(
            update={"strategy_used": doc.strategy_used + "_escalated"}
        )

    def _escalate_to_vision_if_appropriate(
        self, source: Path | str | BinaryIO, profile: DocumentProfile, doc: ExtractedDocument
    ) -> ExtractedDocument:
        if "vision" in doc.strategy_used:
            return doc
        if profile.extraction_cost != ExtractionCost.NEEDS_VISION_MODEL and profile.origin_type != OriginType.SCANNED_IMAGE:
            rcfg = _get_router_config(self._config)
            if not rcfg.get("escalate_to_vision_on_low_confidence", False):
                return doc
        try:
            vision_doc = self._vision.extract(source)
            return vision_doc.model_copy(
                update={"strategy_used": doc.strategy_used + "_vision_escalated"}
            )
        except _VISION_FALLBACK_ERRORS:
            return doc
