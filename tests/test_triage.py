import pytest
from pathlib import Path

from src.agents import TriageAgent
from src.models import (
    DocumentProfile,
    DomainHint,
    ExtractionCost,
    LayoutComplexity,
    OriginType,
)


def test_triage_returns_document_profile(sample_pdf_path: Path) -> None:
    agent = TriageAgent()
    profile = agent.profile(sample_pdf_path)
    assert isinstance(profile, DocumentProfile)
    assert profile.origin_type in (
        OriginType.NATIVE_DIGITAL,
        OriginType.SCANNED_IMAGE,
        OriginType.MIXED,
        OriginType.FORM_FILLABLE,
    )
    assert profile.layout_complexity in (
        LayoutComplexity.SINGLE_COLUMN,
        LayoutComplexity.MULTI_COLUMN,
        LayoutComplexity.TABLE_HEAVY,
        LayoutComplexity.FIGURE_HEAVY,
        LayoutComplexity.MIXED,
    )
    assert profile.extraction_cost in (
        ExtractionCost.FAST_TEXT_SUFFICIENT,
        ExtractionCost.NEEDS_LAYOUT_MODEL,
        ExtractionCost.NEEDS_VISION_MODEL,
    )
    assert profile.domain_hint in (
        DomainHint.FINANCIAL,
        DomainHint.LEGAL,
        DomainHint.TECHNICAL,
        DomainHint.MEDICAL,
        DomainHint.GENERAL,
    )
    assert hasattr(profile, "domain_id") and isinstance(profile.domain_id, str)
    assert hasattr(profile.language, "code") and hasattr(profile.language, "confidence")


def test_triage_accepts_config() -> None:
    agent = TriageAgent(config={"triage": {"char_threshold": 50, "table_heavy_threshold": 10}})
    assert agent.char_threshold == 50
    assert agent.table_heavy_threshold == 10


def test_triage_raises_on_missing_file() -> None:
    agent = TriageAgent()
    with pytest.raises(FileNotFoundError):
        agent.profile(Path("/nonexistent.pdf"))
