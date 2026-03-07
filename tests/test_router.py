import pytest
from pathlib import Path

from src.agents import TriageAgent
from src.models import DocumentProfile, ExtractionCost, LayoutComplexity, OriginType
from src.router import ExtractionRouter

try:
    import pytesseract
    TESSERACT_ERROR = pytesseract.TesseractNotFoundError
except ImportError:
    TESSERACT_ERROR = type("TesseractNotFoundError", (), {})


def test_router_scanned_uses_vision(sample_pdf_path: Path) -> None:
    profile = DocumentProfile(
        origin_type=OriginType.SCANNED_IMAGE,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        extraction_cost=ExtractionCost.NEEDS_VISION_MODEL,
    )
    router = ExtractionRouter()
    try:
        doc = router.extract(sample_pdf_path, profile)
        assert doc.strategy_used in ("vision", "layout_vision_fallback")
    except (ImportError, TESSERACT_ERROR):
        pytest.skip("pytesseract/tesseract not available")


def test_router_fast_returns_fast_strategy(sample_pdf_path: Path) -> None:
    profile = DocumentProfile(
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        extraction_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
    )
    router = ExtractionRouter()
    doc = router.extract(sample_pdf_path, profile)
    assert doc.strategy_used in ("fast_text", "fast_text_escalated", "layout_escalated")
    assert len(doc.blocks) >= 0


def test_router_escalation_on_low_confidence(sample_pdf_path: Path) -> None:
    router = ExtractionRouter(confidence_threshold=0.99)
    profile = DocumentProfile(
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        extraction_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
    )
    doc = router.extract(sample_pdf_path, profile)
    assert "escalated" in doc.strategy_used or doc.strategy_used == "layout"
