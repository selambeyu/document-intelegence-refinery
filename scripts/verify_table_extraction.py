"""
Verify table extraction against a ground-truth set. Computes table-level
precision, recall, and F1. Run after placing PDFs in data_dir (doc_id = stem).

Usage:
  uv run python scripts/verify_table_extraction.py [path_to_gold] [data_dir]
  Default gold: tests/fixtures/ground_truth_tables.json
  Default data_dir: data (project root). PDFs must be named {doc_id}.pdf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.agents import TriageAgent
from src.router import ExtractionRouter
from src.utils import get_refinery_base, load_rules


def _normalize_cells(cells: list) -> list[str]:
    return [str(c).strip().lower() for c in cells]


def _table_match(gold: dict, pred_headers: list[str], pred_rows: list[list[str]]) -> bool:
    gold_headers = gold.get("headers") or []
    gold_rows = gold.get("rows") or []
    if len(pred_headers) != len(gold_headers) or len(pred_rows) != len(gold_rows):
        return False
    g_h = _normalize_cells(gold_headers)
    p_h = _normalize_cells(pred_headers)
    if g_h != p_h:
        return False
    for gr, pr in zip(gold_rows, pred_rows):
        if _normalize_cells(gr) != _normalize_cells(pr):
            return False
    return True


def _find_pdf(data_dir: Path, doc_id: str) -> Path | None:
    candidates = [
        data_dir / f"{doc_id}.pdf",
        data_dir / f"{doc_id}.PDF",
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in data_dir.rglob("*.pdf"):
        if p.stem == doc_id:
            return p
    return None


def run_verification(
    gold_path: Path,
    data_dir: Path,
    refinery_base: Path | None = None,
) -> dict:
    gold_raw = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_tables = [g for g in gold_raw if isinstance(g, dict) and "doc_id" in g and "page" in g]

    rules = load_rules()
    triage = TriageAgent(config=rules)
    router = ExtractionRouter(
        confidence_threshold=float(rules.get("router", {}).get("confidence_escalation_threshold", 0.6)),
        config=rules,
    )

    doc_ids = list({g["doc_id"] for g in gold_tables})
    doc_id_to_pdf: dict[str, Path] = {}
    for d in doc_ids:
        pdf = _find_pdf(data_dir, d)
        if pdf:
            doc_id_to_pdf[d] = pdf

    predicted_by_doc: dict[str, list[tuple[int, list[str], list[list[str]]]]] = {}
    for doc_id, pdf_path in doc_id_to_pdf.items():
        profile = triage.profile(pdf_path)
        doc = router.extract(pdf_path, profile)
        preds = []
        for t in doc.tables:
            page = t.page or (t.bbox.page if t.bbox else 0)
            preds.append((page, t.headers, t.rows))
        preds.sort(key=lambda x: (x[0], len(x[1]), len(x[2])))
        predicted_by_doc[doc_id] = preds

    tp = 0
    matched_gold = set()
    for i, gold in enumerate(gold_tables):
        doc_id = gold["doc_id"]
        page = int(gold.get("page", 0))
        preds = predicted_by_doc.get(doc_id, [])
        for p_page, p_headers, p_rows in preds:
            if p_page != page:
                continue
            if _table_match(gold, p_headers, p_rows):
                tp += 1
                matched_gold.add(i)
                break

    n_gold = len(gold_tables)
    n_pred = sum(len(p) for p in predicted_by_doc.values())
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_gold_tables": n_gold,
        "n_predicted_tables": n_pred,
        "true_positives": tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "docs_with_pdf": len(doc_id_to_pdf),
        "docs_in_gold": len(doc_ids),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    gold_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "tests" / "fixtures" / "ground_truth_tables.json"
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "data"
    refinery_base = get_refinery_base() if len(sys.argv) <= 2 else None

    if not gold_path.exists():
        print("Gold file not found:", gold_path, file=sys.stderr)
        sys.exit(1)
    if not data_dir.exists():
        print("Data dir not found:", data_dir, file=sys.stderr)
        sys.exit(1)

    out = run_verification(gold_path, data_dir, refinery_base)
    print("Table extraction verification (ground truth)")
    print("  Gold tables:", out["n_gold_tables"])
    print("  Predicted tables:", out["n_predicted_tables"])
    print("  True positives:", out["true_positives"])
    print("  Precision:", out["precision"])
    print("  Recall:", out["recall"])
    print("  F1:", out["f1"])

    docs_dir = repo / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = docs_dir / "table_extraction_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Metrics saved to", metrics_path)


if __name__ == "__main__":
    main()
