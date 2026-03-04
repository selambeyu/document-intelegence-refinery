"""Load extraction_rules.yaml; fallback to defaults if missing or no PyYAML."""
from pathlib import Path

DEFAULTS = {
    "triage": {
        "char_threshold": 100,
        "image_area_ratio_threshold": 0.5,
        "table_heavy_threshold": 3,
        "figure_heavy_ratio": 0.4,
    },
    "router": {"confidence_escalation_threshold": 0.6},
}


def load_rules(path: Path | str | None = None) -> dict:
    path = path or Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
    path = Path(path)
    if not path.exists():
        return DEFAULTS
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or DEFAULTS
    except ImportError:
        return DEFAULTS
