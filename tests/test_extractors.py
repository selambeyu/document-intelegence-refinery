import pytest
from pathlib import Path

from src.extractors import FastTextExtractor, LayoutExtractor
from src.models import ExtractedDocument


def test_fast_extractor_returns_schema(sample_pdf_path: Path) -> None:
    ext = FastTextExtractor()
    doc = ext.extract(sample_pdf_path)
    assert isinstance(doc, ExtractedDocument)
    assert doc.strategy_used == "fast_text"
    assert 0 <= doc.confidence <= 1
    assert isinstance(doc.blocks, list)
    for b in doc.blocks:
        assert b.bbox.page >= 1
        assert b.text is not None


def test_layout_extractor_returns_schema(sample_pdf_path: Path) -> None:
    ext = LayoutExtractor()
    doc = ext.extract(sample_pdf_path)
    assert isinstance(doc, ExtractedDocument)
    assert "layout" in doc.strategy_used
    assert doc.confidence >= 0.8
    assert isinstance(doc.blocks, list)
