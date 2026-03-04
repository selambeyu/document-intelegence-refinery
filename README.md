# Document Intelligence Refinery

Adaptive document extraction pipeline: **PDF → Triage → Router → Extraction → Normalized output**.

## Setup

Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

Or install in editable mode without lockfile: `uv pip install -e .`

To run without activating the venv: `uv run refinery path/to/document.pdf` or `uv run python -m src.pipelines.run_pipeline path/to/document.pdf`

Optional: install Tesseract OCR for vision strategy (scanned PDFs):

- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr`

## Run

From project root:

```bash
python -m src.pipelines.run_pipeline path/to/document.pdf
```

Or after install:

```bash
refinery path/to/document.pdf
```

Output: strategy used, number of blocks, confidence.

Profiles and extraction ledger are written under `.refinery/` (profiles as JSON, ledger as JSONL). Disable with `save_artifacts=False` in `run_pipeline()`.

## Config

`rubric/extraction_rules.yaml` — triage thresholds (char_threshold, image_area_ratio_threshold, table_heavy_threshold) and router `confidence_escalation_threshold`. Omit the file or PyYAML to use built-in defaults.

## Artifacts

- `.refinery/profiles/{doc_id}.json` — `DocumentProfile` per document (doc_id = path stem).
- `.refinery/logs/extraction_ledger.jsonl` — one line per run: doc_id, strategy_used, confidence, block_count.

## Ingestion

`src.utils.collect_pdfs(path)` — returns a list of PDF paths for a file or directory. Use for batch runs.

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

**If you see `ModuleNotFoundError: No module named 'pdfplumber'`:** tests are using a Python environment that doesn’t have the project dependencies.

- **With uv:** ensure you run tests via uv so it uses the project venv:  
  `uv sync` (or `uv sync --extra dev` for pytest), then `uv run pytest tests/ -v`
- **With pip/conda:** install the project in the same environment you use for pytest:  
  `pip install -e .`  
  Or install at least: `pip install pdfplumber pymupdf pytesseract Pillow pydantic pyyaml langdetect fast-langdetect`

If you use a project `.venv` created by conda, uv may fail to read it. Remove `.venv` and run `uv sync` to let uv create a new venv, or run pytest with that conda env after `pip install -e .`.

## Compare extractors (pdfplumber, Docling, MinerU)

Run all three on the same PDF(s) for Phase 0 / quality comparison:

```bash
uv sync --extra compare    # install Docling
uv run python scripts/run_all_extractors.py data/
# Or single file:  uv run python scripts/run_all_extractors.py "data/Audit Report - 2023.pdf"
# Skip MinerU:     uv run python scripts/run_all_extractors.py data/ --no-mineru
# Skip Docling:     uv run python scripts/run_all_extractors.py data/ --no-docling
```

Output under `.refinery/compare_out/`: `*_pdfplumber.json`, `*_docling.md`, `mineru/` (MinerU needs `pip install mineru` separately).

## Structure

- `src/models/` — Pydantic schemas (`DocumentProfile`, `ExtractedDocument`, etc.)
- `src/agents/` — Triage agent (pdfplumber heuristics)
- `src/extractors/` — Fast (pdfplumber), Layout (pdfplumber), Vision (pymupdf + pytesseract)
- `src/router/` — Strategy selection and confidence-gated escalation to Layout
- `src/utils/` — `ensure_refinery_dirs`, `get_refinery_base`, `load_rules` (from rubric), `collect_pdfs`

- `src/pipelines/run_pipeline.py` — profile → extract → print; writes to `.refinery/` when `save_artifacts=True`

## Router rules

- **Scanned** → Vision (OCR)
- **Simple** (digital) → Fast
- Else → Layout  
- If confidence &lt; 0.6 → escalate to Layout
