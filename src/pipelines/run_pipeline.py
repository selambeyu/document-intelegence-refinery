import json
import sys
import time
from pathlib import Path

from src.agents import TriageAgent
from src.router import ExtractionRouter
from src.utils import ensure_refinery_dirs, get_refinery_base, load_rules

COST_BY_STRATEGY = {
    "fast_text": 0.0,
    "layout": 0.01,
    "layout_docling": 0.01,
    "vision": 0.05,
    "vision_ocr": 0.0,
    "vision_ocr_fallback": 0.0,
    "fast_text_escalated": 0.01,
    "layout_vision_fallback": 0.05,
}


def _cost_estimate(strategy_used: str) -> float:
    base = strategy_used.split("_")[0] if strategy_used else ""
    return COST_BY_STRATEGY.get(strategy_used) or COST_BY_STRATEGY.get(base, 0.01)


def run_pipeline(
    document_path: str | Path,
    refinery_base: Path | None = None,
    save_artifacts: bool = True,
) -> None:
    path = Path(document_path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    base = refinery_base or get_refinery_base()
    ensure_refinery_dirs(base)
    rules = load_rules()
    triage = TriageAgent(config=rules)
    router = ExtractionRouter(
        confidence_threshold=float(rules.get("router", {}).get("confidence_escalation_threshold", 0.6))
    )
    profile = triage.profile(path)
    doc_id = path.stem
    if save_artifacts:
        (base / "profiles" / f"{doc_id}.json").write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )
    print("Profile:", profile.origin_type.value, profile.layout_complexity.value,
          profile.domain_hint.value, profile.extraction_cost.value,
          f"lang={profile.language.code}({profile.language.confidence:.2f})")
    t0 = time.perf_counter()
    doc = router.extract(path, profile)
    elapsed = time.perf_counter() - t0
    if save_artifacts:
        ledger_path = base / "logs" / "extraction_ledger.jsonl"
        entry = {
            "doc_id": doc_id,
            "strategy_used": doc.strategy_used,
            "confidence": doc.confidence,
            "block_count": len(doc.blocks),
            "processing_time": round(elapsed, 4),
            "cost_estimate": _cost_estimate(doc.strategy_used),
        }
        if getattr(doc, "cost_actual", None) is not None:
            entry["cost_actual"] = doc.cost_actual
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("Strategy used:", doc.strategy_used)
    print("Number of blocks:", len(doc.blocks))
    print("Confidence:", doc.confidence)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: refinery <path-to-pdf>", file=sys.stderr)
        sys.exit(1)
    run_pipeline(sys.argv[1])
