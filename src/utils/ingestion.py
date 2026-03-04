"""Collect PDF paths from a file or directory."""
from pathlib import Path


def collect_pdfs(path: Path | str) -> list[Path]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    if p.is_file():
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {p}")
        return [p]
    return sorted(p.glob("**/*.pdf"))
