# Document Intelligence Refinery

Adaptive document extraction pipeline: **PDF → Triage → Router → Extraction → Chunking → PageIndex → Vector store → Query agent**.

**Requires Python 3.10+.** The steps below enable deployment in under 10 minutes.

## Quick start 

1. **Install** (from project root):
   ```bash
   uv venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   uv sync
   ```
   Or with pip: `pip install -e .`

2. **Run extraction** on a PDF or folder:
   ```bash
   refinery path/to/document.pdf
   # or: refinery path/to/folder
   ```

3. **Optional — query**: Start the web UI and ask questions:
   ```bash
   refinery-serve
   ```
   Then open **http://localhost:8000** to upload PDFs and run Q&A.

No API key is required for extraction-only. For query and VLM extraction, set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (see [Environment variables](#environment-variables)).

---

## How to run

**1. Install (from project root)**

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

Or with pip: `pip install -e .`

**2a. CLI — single PDF or batch**

```bash
refinery path/to/document.pdf
refinery path/to/folder   # processes all PDFs in folder (and subfolders)
```

Or: `uv run python -m src.pipelines.run_pipeline path/to/document.pdf`

**2b. Web UI (for non-technical users)**

```bash
refinery-serve
```

If `refinery-serve` is not found, reinstall the project then try again (`pip install -e .` or `uv sync`), or run:

```bash
uv run python scripts/serve.py
```

Then open **http://localhost:8000**. You can:
- **Upload a PDF** — processes it and adds it to the index.
- **Ask a question** — optional document filter; returns answer and source citations (document, page).

Query uses `OPENAI_API_KEY` or `OPENROUTER_API_KEY` by default; or use **Ollama (local)** — no API key needed.

**3. (Optional) Query from command line**

After processing at least one document:

```bash
uv sync --extra rag   # for ChromaDB semantic search
uv run python scripts/example_qa.py .refinery "Your question" [doc_id]
```

**Environment variables (API keys)**

| Variable | Used for | Notes |
|----------|-----------|--------|
| `OPENAI_API_KEY` | Query agent (LangGraph), Vision extractor, PageIndex LLM summaries | OpenAI API key |
| `OPENROUTER_API_KEY` | Same as above, with OpenRouter base URL | If set, vision/query use `https://openrouter.ai/api/v1` |
| `USE_OLLAMA` or `OLLAMA_USE` | Switch to local Ollama | Set to `1` to use Ollama; no API key needed |
| `OLLAMA_MODEL` | Model name for Ollama | Default: `llama3.2` |

Either key is enough; both are checked (`OPENROUTER` first, then `OPENAI`). No key is required for extraction-only runs (fast/layout/OCR fallback). Copy `.env.example` to `.env` in the project root and add your key; the app loads `.env` automatically via `python-dotenv`.

**Local mode (Ollama) — no API credits**

Use a local model instead of OpenAI/OpenRouter:

```bash
# 1. Install Ollama and pull a model
# https://ollama.ai — then: ollama pull llama3.2

# 2. Enable local mode
export USE_OLLAMA=1
# Optional: OLLAMA_MODEL=llama3.2  (default)

# 3. Run query (CLI or Web UI)
refinery-serve
# or: uv run python scripts/example_qa.py .refinery "Your question"
```

`USE_OLLAMA=1` (or `OLLAMA_USE=1`) switches the Query agent and PageIndex LLM summaries to Ollama. No API key required.

**API (for integrations)**

- `GET /documents` — list processed document IDs.
- `POST /documents` — upload a PDF (multipart form, field `file`); runs the full pipeline, returns `doc_id`.
- `POST /query` — body `{"question": "...", "doc_ids": ["id1"]}` (optional); returns `{"answer": "...", "provenance": [...]}`.

Run the server: `refinery-serve` or `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`. Optional env: `REFINERY_BASE` = project root (directory containing `.refinery`).

**4.  Tesseract for scanned PDFs**

- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr`

---

## Setup (details)

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

---

## Run (alternatives)

From project root:

```bash
python -m src.pipelines.run_pipeline path/to/document.pdf
```

Or after install:

```bash
refinery path/to/document.pdf
```

Output: strategy used, number of blocks, confidence, LDUs, PageIndex, facts.

When `save_artifacts=True` (default), the pipeline writes profiles, ledger, PageIndex, vector store, and fact table under `.refinery/`. Disable with `save_artifacts=False`.

## Config

`rubric/extraction_rules.yaml` — triage thresholds (char_threshold, image_area_ratio_threshold, table_heavy_threshold) and router `confidence_escalation_threshold`. Add new document types by adding keys under `domain_keywords` and optionally under `extract_fact_domains`; no code change required. See `docs/FDE_READINESS.md`.

## Artifacts

- `.refinery/profiles/{doc_id}.json` — `DocumentProfile` per document.
- `.refinery/logs/extraction_ledger.jsonl` — one line per run: doc_id, strategy_used, confidence, block_count.
- `.refinery/pageindex/{doc_id}.json` — PageIndex tree.
- `.refinery/vector_store/` — ChromaDB (optional) for LDU search.
- `.refinery/facts.db` — SQLite fact table.

## Ingestion

Single file or folder: `refinery path/to/file.pdf` or `refinery path/to/folder`. Programmatic: `src.utils.collect_pdfs(path)` returns a list of PDF paths for a file or directory.

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

## Table verification and PageIndex benchmark

- **Table extraction (ground truth):** Put PDFs in `data/` and run  
  `uv run python scripts/verify_table_extraction.py [path_to_gold] [data_dir]`  
  Default gold: `tests/fixtures/ground_truth_tables.json`. Metrics are written to `docs/table_extraction_metrics.json`.
- **PageIndex vs vector search:** After processing documents, run  
  `uv run python scripts/benchmark_pageindex_vs_vector.py [.refinery]`  
  Results go to `docs/pageindex_benchmark_results.json`. See `docs/PROVENANCE.md` for the interpretation.

## Structure

- `src/models/` — Pydantic schemas (`DocumentProfile`, `ExtractedDocument`, etc.)
- `src/agents/` — Triage agent (pdfplumber heuristics)
- `src/extractors/` — Fast (pdfplumber), Layout (pdfplumber), Vision (pymupdf + pytesseract)
- `src/router/` — Strategy selection and confidence-gated escalation to Layout
- `src/utils/` — `ensure_refinery_dirs`, `get_refinery_base`, `load_rules` (from rubric), `collect_pdfs`

- `src/pipelines/run_pipeline.py` — triage → extract → chunk → PageIndex → vector store → facts; writes to `.refinery/` when `save_artifacts=True`.

## Router rules

- **Scanned** → Vision (OCR)
- **Simple** (digital) → Fast
- Else → Layout  
- If confidence &lt; 0.6 → escalate to Layout
