import json
import sys
import time
from pathlib import Path

from src.agents import (
    ChunkingEngine,
    ChunkValidator,
    TriageAgent,
    build_pageindex,
    extract_and_store_facts,
    save_pageindex,
)
from src.router import ExtractionRouter
from src.utils import (
    ensure_refinery_dirs,
    get_refinery_base,
    load_rules,
    VectorStore,
)
from src.utils.fact_store import FactStore

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
        confidence_threshold=float(rules.get("router", {}).get("confidence_escalation_threshold", 0.6)),
        config=rules,
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
        if getattr(doc, "review_flag", False):
            entry["review_flag"] = True
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("Strategy used:", doc.strategy_used)
    print("Number of blocks:", len(doc.blocks))
    print("Confidence:", doc.confidence)
    if getattr(doc, "review_flag", False):
        print("Review flagged: low confidence — consider human review")

    if save_artifacts:
        chunk_cfg = rules.get("chunking", {})
        engine = ChunkingEngine(config=chunk_cfg)
        ldus = engine.chunk(doc)
        validator = ChunkValidator(max_tokens=chunk_cfg.get("max_tokens", 512))
        valid_ldus = validator.filter_valid(ldus)
        if len(valid_ldus) < len(ldus):
            errors = validator.validate(ldus)
            print("ChunkValidator: dropped", len(ldus) - len(valid_ldus), "invalid LDU(s).", errors[:5])
        ldus = valid_ldus
        index = build_pageindex(doc, ldus, use_llm_summary=False)
        pageindex_path = base / "pageindex" / f"{doc_id}.json"
        existing_has_sections = False
        if pageindex_path.exists() and len(ldus) == 0:
            try:
                from src.agents.indexer import load_pageindex
                existing = load_pageindex(pageindex_path)
                existing_has_sections = len(existing.sections) > 0
            except Exception:
                pass
        if not existing_has_sections or len(ldus) > 0:
            save_pageindex(index, pageindex_path)
        elif existing_has_sections:
            print("Keeping existing PageIndex (current run produced 0 LDUs; previous extraction had content).")
        vs = VectorStore(persist_directory=base / "vector_store")
        if ldus:
            vs.add_document(doc_id, ldus)
        elif existing_has_sections:
            print("Skipping vector store update (0 LDUs; keeping previous chunks).")
        store = FactStore(db_path=base / "facts.db")
        n_facts = extract_and_store_facts(doc_id, doc, ldus, profile, store, config=rules)
        print("LDUs:", len(ldus), "PageIndex saved." if not (existing_has_sections and len(ldus) == 0) else "PageIndex preserved.", "Facts stored:", n_facts)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: refinery <path-to-pdf> | refinery <path-to-folder>  (processes all PDFs in folder)", file=sys.stderr)
        sys.exit(1)
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        sys.exit(1)
    base = get_refinery_base()
    if target.is_file():
        if target.suffix.lower() != ".pdf":
            print("Error: not a PDF file.", file=sys.stderr)
            sys.exit(1)
        run_pipeline(target, refinery_base=base)
    else:
        from src.utils.ingestion import collect_pdfs
        try:
            pdfs = collect_pdfs(target)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if not pdfs:
            print(f"No PDFs found under {target}", file=sys.stderr)
            sys.exit(1)
        print(f"Processing {len(pdfs)} PDF(s)...")
        for i, path in enumerate(pdfs, 1):
            print(f"[{i}/{len(pdfs)}] {path.name}")
            run_pipeline(path, refinery_base=base)
        print("Batch done.")
