import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture
def sample_pdf_path() -> Path:
    p = DATA_DIR / "Audit Report - 2023.pdf"
    if not p.exists():
        pytest.skip("data/Audit Report - 2023.pdf not found")
    return p
