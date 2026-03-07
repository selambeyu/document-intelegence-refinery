from pathlib import Path
from typing import BinaryIO, Optional

import pdfplumber

from src.agents.domain_hint import DomainHintClassifier, KeywordDomainHintClassifier
from src.models import (
    DetectedLanguage,
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LayoutComplexity,
    OriginType,
)


class TriageAgent:
    def __init__(
        self,
        config: dict | None = None,
        domain_hint_classifier: DomainHintClassifier | None = None,
    ):
        cfg = (config or {}).get("triage", {})
        self.char_threshold = cfg.get("char_threshold", 100)
        self.image_area_ratio_threshold = cfg.get("image_area_ratio_threshold", 0.5)
        self.table_heavy_threshold = cfg.get("table_heavy_threshold", 3)
        self.figure_heavy_ratio = cfg.get("figure_heavy_ratio", 0.4)
        self.domain_hint_classifier = domain_hint_classifier or KeywordDomainHintClassifier(config=config)

    def profile(self, source: Path | str | BinaryIO) -> DocumentProfile:
        path = None
        if isinstance(source, (Path, str)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {path}")
        with pdfplumber.open(source) as pdf:
            origin_type = self._detect_origin(pdf)
            layout_complexity = self._detect_layout(pdf)
            language = self._detect_language(pdf)
            domain_hint = self._detect_domain_hint(pdf)
            extraction_cost = self._infer_extraction_cost(origin_type, layout_complexity)
        return DocumentProfile(
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language=language,
            domain_hint=domain_hint,
            extraction_cost=extraction_cost,
        )

    def _detect_origin(self, pdf: pdfplumber.PDF) -> OriginType:
        if self._has_acroform(pdf):
            return OriginType.FORM_FILLABLE
        total_chars = 0
        total_area = 0.0
        image_area = 0.0
        chars_with_font = 0
        for page in pdf.pages:
            total_area += float(page.width * page.height)
            chars = page.chars
            if chars:
                total_chars += len(chars)
                chars_with_font += sum(1 for c in chars if c.get("fontname"))
            if page.images:
                for im in page.images:
                    w = im.get("width") or 0
                    h = im.get("height") or 0
                    image_area += w * h
        if total_area <= 0:
            return OriginType.SCANNED_IMAGE
        image_ratio = image_area / total_area if total_area else 0
        font_ratio = chars_with_font / total_chars if total_chars else 0.0
        if total_chars < self.char_threshold and image_ratio > self.image_area_ratio_threshold:
            return OriginType.SCANNED_IMAGE
        if total_chars >= self.char_threshold and image_ratio < self.image_area_ratio_threshold:
            return OriginType.NATIVE_DIGITAL
        if font_ratio >= 0.6 and image_ratio < self.image_area_ratio_threshold:
            return OriginType.NATIVE_DIGITAL
        if font_ratio < 0.2 and image_ratio > self.image_area_ratio_threshold:
            return OriginType.SCANNED_IMAGE
        return OriginType.MIXED

    def _has_acroform(self, pdf: pdfplumber.PDF) -> bool:
        try:
            doc = getattr(pdf, "doc", None)
            if doc is None:
                return False
            catalog = getattr(doc, "catalog", None)
            if catalog is None:
                return False
            if hasattr(catalog, "get"):
                return catalog.get("AcroForm") is not None or catalog.get("/AcroForm") is not None
            return False
        except Exception:
            return False

    def _detect_layout(self, pdf: pdfplumber.PDF) -> LayoutComplexity:
        table_count = 0
        total_area = 0.0
        image_area = 0.0
        for page in pdf.pages:
            total_area += float(page.width * page.height)
            tables = page.find_tables()
            if tables:
                table_count += len(list(tables))
            if page.images:
                for im in page.images:
                    image_area += (im.get("width") or 0) * (im.get("height") or 0)
        if total_area <= 0:
            return LayoutComplexity.MIXED
        image_ratio = image_area / total_area
        if table_count > self.table_heavy_threshold:
            return LayoutComplexity.TABLE_HEAVY
        if image_ratio >= self.figure_heavy_ratio:
            return LayoutComplexity.FIGURE_HEAVY
        if self._looks_multi_column(pdf):
            return LayoutComplexity.MULTI_COLUMN
        return LayoutComplexity.SINGLE_COLUMN

    def _looks_multi_column(self, pdf: pdfplumber.PDF) -> bool:
        page_widths = [float(p.width) for p in pdf.pages if p.width]
        if not page_widths:
            return False
        mid = sum(page_widths) / len(page_widths) / 2
        left_count = right_count = 0
        for page in pdf.pages:
            chars = page.chars or []
            for c in chars:
                x0 = c.get("x0", 0)
                if x0 < mid:
                    left_count += 1
                else:
                    right_count += 1
        total = left_count + right_count
        if total < 50:
            return False
        left_ratio = left_count / total
        return 0.25 <= left_ratio <= 0.75

    def _detect_language(self, pdf: pdfplumber.PDF) -> DetectedLanguage:
        samples = []
        for page in pdf.pages[:5]:
            text = page.extract_text()
            if text:
                samples.append(text[:2000])
        combined = " ".join(samples).strip()
        if not combined:
            return DetectedLanguage(code="und", confidence=0.0)
        detected = self._detect_language_from_text(combined)
        if detected:
            return detected
        if self._is_amharic_script(combined):
            return DetectedLanguage(code="am", confidence=0.75)
        return DetectedLanguage(code="und", confidence=0.0)

    def _detect_language_from_text(self, text: str) -> Optional[DetectedLanguage]:
        try:
            from fast_langdetect import detect
            result = detect(text, model="lite", k=1)
            if result and len(result) > 0:
                item = result[0]
                if isinstance(item, dict):
                    code = (item.get("lang") or "").replace("__label__", "")
                    conf = float(item.get("score") or 0)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    code, conf = str(item[0]).replace("__label__", ""), float(item[1])
                elif hasattr(item, "lang") and hasattr(item, "score"):
                    code, conf = str(getattr(item, "lang")).replace("__label__", ""), float(getattr(item, "score"))
                else:
                    code, conf = str(item).replace("__label__", ""), 0.8
                if code and conf > 0.1:
                    return DetectedLanguage(code=code, confidence=min(conf, 1.0))
        except Exception:
            pass
        try:
            import langdetect
            langdetect.DetectorFactory.seed = 0
            result = langdetect.detect_langs(text)
            if result:
                top = result[0]
                if top.lang != "und" and top.prob > 0.1:
                    return DetectedLanguage(code=top.lang, confidence=float(top.prob))
        except Exception:
            pass
        return None

    def _is_amharic_script(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        ethiopic = sum(1 for c in text if "\u1200" <= c <= "\u137f" or "\uab00" <= c <= "\uab2f")
        ratio = ethiopic / len(text)
        return ratio >= 0.15

    def _detect_domain_hint(self, pdf: pdfplumber.PDF) -> DomainHint:
        samples = []
        for page in pdf.pages[:10]:
            text = page.extract_text()
            if text:
                samples.append(text.lower())
        combined = " ".join(samples)
        return self.domain_hint_classifier.classify(combined)

    def _infer_extraction_cost(
        self, origin: OriginType, layout: LayoutComplexity
    ) -> ExtractionCost:
        if origin == OriginType.SCANNED_IMAGE:
            return ExtractionCost.NEEDS_VISION_MODEL
        if origin == OriginType.FORM_FILLABLE:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
        if layout == LayoutComplexity.SINGLE_COLUMN and origin == OriginType.NATIVE_DIGITAL:
            return ExtractionCost.FAST_TEXT_SUFFICIENT
        if layout in (LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN, LayoutComplexity.FIGURE_HEAVY):
            return ExtractionCost.NEEDS_LAYOUT_MODEL
        if layout == LayoutComplexity.MIXED:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
        if origin == OriginType.MIXED:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
        return ExtractionCost.NEEDS_LAYOUT_MODEL
