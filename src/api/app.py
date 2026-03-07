"""
Refinery web API and minimal UI.
Run: uvicorn src.api.app:app --reload
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.pipelines.run_pipeline import run_pipeline
from src.utils import ensure_refinery_dirs, get_refinery_base
from src.utils.api_key import ApiKeyRequiredError
from src.utils.fact_store import FactStore
from src.utils.vector_store import VectorStore

app = FastAPI(title="Document Intelligence Refinery", version="0.1.0")

REFINERY_BASE = Path(os.environ.get("REFINERY_BASE", "")).resolve() if os.environ.get("REFINERY_BASE") else None


def _base() -> Path:
    return get_refinery_base(REFINERY_BASE) if REFINERY_BASE else get_refinery_base()


@app.get("/documents")
def list_documents():
    base = _base()
    profiles_dir = base / "profiles"
    if not profiles_dir.exists():
        return {"doc_ids": []}
    doc_ids = [p.stem for p in profiles_dir.glob("*.json")]
    return {"doc_ids": sorted(doc_ids)}


@app.post("/documents")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed.")
    base = _base()
    base.mkdir(parents=True, exist_ok=True)
    ensure_refinery_dirs(base)
    uploads = base / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ") or "document"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    path = uploads / safe_name
    try:
        content = await file.read()
        path.write_bytes(content)
        await asyncio.to_thread(run_pipeline, path, refinery_base=base)
        doc_id = path.stem
        return {"status": "ok", "doc_id": doc_id}
    except Exception as e:
        if path.exists():
            path.unlink(missing_ok=True)
        raise HTTPException(500, str(e))


class QueryBody(BaseModel):
    question: str
    doc_ids: list[str] | None = None


@app.post("/query")
def query(body: QueryBody):
    base = _base()
    try:
        from src.agents import QueryAgent
        vs = VectorStore(persist_directory=base / "vector_store")
        store = FactStore(db_path=base / "facts.db")
        agent = QueryAgent(
            pageindex_dir=base / "pageindex",
            vector_store=vs,
            fact_store=store,
        )
        answer, chain = agent.query(body.question, doc_ids=body.doc_ids, k=10)
        return {
            "answer": answer,
            "provenance": [c.model_dump() for c in chain.citations],
        }
    except ApiKeyRequiredError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        msg = str(e)
        if "401" in msg or "User not found" in msg or "Invalid" in msg.lower() or "authentication" in msg.lower() or "API key" in msg:
            raise HTTPException(
                401,
                "Invalid or expired API key. Check OPENAI_API_KEY or OPENROUTER_API_KEY in .env. "
                "Get a key from platform.openai.com or openrouter.ai.",
            )
        raise HTTPException(500, msg)


@app.get("/", response_class=HTMLResponse)
def ui():
    return _HTML


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Intelligence Refinery</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto; padding: 1.5rem; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .sub { color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
    section { margin-bottom: 2rem; }
    section h2 { font-size: 1.1rem; margin-bottom: 0.5rem; }
    input[type="file"], input[type="text"], button { margin-right: 0.5rem; margin-bottom: 0.5rem; }
    button { padding: 0.4rem 0.8rem; cursor: pointer; background: #111; color: #fff; border: none; border-radius: 4px; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #uploadStatus, #queryResult, #queryProvenance { margin-top: 0.5rem; padding: 0.75rem; background: #f5f5f5; border-radius: 4px; font-size: 0.9rem; white-space: pre-wrap; }
    .err { background: #fee !important; color: #c00; }
    .citations { font-size: 0.85rem; color: #555; margin-top: 0.5rem; }
    select { padding: 0.4rem; margin-bottom: 0.5rem; min-width: 200px; }
  </style>
</head>
<body>
  <h1>Document Intelligence Refinery</h1>
  <p class="sub">Upload PDFs and ask questions. Answers are based only on your uploaded and processed documents (with source citations).</p>

  <section>
    <h2>1. Upload a PDF</h2>
    <input type="file" id="file" accept=".pdf">
    <button id="uploadBtn">Upload & process</button>
    <div id="uploadStatus"></div>
  </section>

  <section>
    <h2>2. Ask a question</h2>
    <p>Answers are based <strong>only</strong> on the documents listed below (your processed PDFs).</p>
    <p>
      <select id="docIds" multiple></select>
      <span style="font-size:0.85rem;color:#666"> Leave none selected to search all processed documents; select one or more to limit the search.</span>
    </p>
    <p>
      <input type="text" id="question" placeholder="Your question..." style="width:100%;max-width:400px;padding:0.5rem;">
      <button id="queryBtn">Ask</button>
    </p>
    <div id="queryResult"></div>
    <div id="queryProvenance" class="citations"></div>
  </section>

  <script>
    const api = (path, opts = {}) => fetch(path, { ...opts, headers: { "Content-Type": "application/json", ...opts.headers } });
    function el(id) { return document.getElementById(id); }
    function show(elId, text, isErr) {
      const el = document.getElementById(elId);
      el.textContent = text;
      el.className = isErr ? "err" : "";
    }

    async function loadDocs() {
      const r = await api("/documents");
      const d = await r.json();
      const sel = el("docIds");
      sel.innerHTML = "";
      (d.doc_ids || []).forEach(id => {
        const o = document.createElement("option");
        o.value = id;
        o.textContent = id;
        sel.appendChild(o);
      });
    }
    loadDocs();

    el("uploadBtn").onclick = async () => {
      const file = el("file").files[0];
      if (!file) { show("uploadStatus", "Choose a PDF file.", true); return; }
      show("uploadStatus", "Uploading...");
      el("uploadBtn").disabled = true;
      try {
        const fd = new FormData();
        fd.append("file", file);
        const r = await fetch("/documents", { method: "POST", body: fd });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || r.statusText);
        show("uploadStatus", "Done. Document ID: " + d.doc_id);
        loadDocs();
      } catch (e) {
        show("uploadStatus", "Error: " + e.message, true);
      }
      el("uploadBtn").disabled = false;
    };

    el("queryBtn").onclick = async () => {
      const question = el("question").value.trim();
      if (!question) { show("queryResult", "Enter a question.", true); return; }
      const sel = el("docIds");
      const docIds = Array.from(sel.selectedOptions).map(o => o.value);
      const allDocIds = Array.from(sel.options).map(o => o.value);
      if (allDocIds.length === 0) {
        show("queryResult", "No processed documents yet. Upload and process at least one PDF in step 1, then try again.", true);
        return;
      }
      show("queryResult", "Searching your documents...");
      el("queryResult").className = "";
      el("queryProvenance").textContent = "";
      el("queryBtn").disabled = true;
      try {
        const r = await api("/query", { method: "POST", body: JSON.stringify({ question, doc_ids: docIds.length ? docIds : null }) });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || r.statusText);
        show("queryResult", d.answer || "No answer.");
        el("queryProvenance").innerHTML = (d.provenance || []).length
          ? "Sources: " + (d.provenance || []).map(c => c.document_name + " p." + c.page_number).join("; ")
          : "";
      } catch (e) {
        show("queryResult", "Error: " + e.message, true);
      }
      el("queryBtn").disabled = false;
    };
  </script>
</body>
</html>
"""
