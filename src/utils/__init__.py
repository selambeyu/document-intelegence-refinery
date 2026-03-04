from pathlib import Path

from .config import load_rules
from .ingestion import collect_pdfs


def ensure_refinery_dirs(base: Path) -> None:
    (base / "profiles").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)


def get_refinery_base(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root) / ".refinery"
    return Path.cwd() / ".refinery"
