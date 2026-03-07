"""
Refinery web API and minimal UI.
Run: uvicorn src.api.app:app --reload
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
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


def _vector_store_from_sqlite(base: Path, limit: int) -> dict:
    """Read Chroma data from chroma.sqlite3 when Chroma API is not available."""
    path = base / "vector_store" / "chroma.sqlite3"
    if not path.exists():
        return {"available": False, "reason": "chroma.sqlite3 not found", "sample": []}
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT COUNT(*) AS n FROM embeddings")
        total = cur.fetchone()[0]
        cur = conn.execute("""
            SELECT e.id, e.embedding_id,
                   max(CASE WHEN m.key = 'doc_id' THEN m.string_value END) AS doc_id,
                   max(CASE WHEN m.key = 'page_refs' THEN m.string_value END) AS page_refs,
                   max(CASE WHEN m.key = 'page' THEN m.int_value END) AS page,
                   max(CASE WHEN m.key = 'chroma:document' THEN m.string_value END) AS content
            FROM embeddings e
            JOIN embedding_metadata m ON e.id = m.id
            GROUP BY e.id
            ORDER BY e.id
            LIMIT ?
        """, (min(limit, 100),))
        rows = cur.fetchall()
    finally:
        conn.close()
    sample = []
    for r in rows:
        content = (r["content"] or "")[:300].replace("\n", " ") + ("..." if len(r["content"] or "") > 300 else "")
        sample.append({
            "id": r["embedding_id"],
            "doc_id": r["doc_id"],
            "page_refs": r["page_refs"],
            "page": r["page"],
            "content_preview": content,
        })
    return {
        "available": True,
        "source": "sqlite",
        "collection": "refinery_ldus",
        "total_chunks": total,
        "sample": sample,
    }


@app.get("/documents")
def list_documents():
    base = _base()
    profiles_dir = base / "profiles"
    if not profiles_dir.exists():
        return {"doc_ids": []}
    doc_ids = [p.stem for p in profiles_dir.glob("*.json")]
    return {"doc_ids": sorted(doc_ids)}


@app.get("/db/vector")
def db_vector_summary(limit: int = 10):
    """View vector store (Chroma): chunk count and sample chunks with doc_id, page_refs."""
    base = _base()
    vs = VectorStore(persist_directory=base / "vector_store")
    if vs._collection is not None:
        coll = vs._collection
        count = coll.count()
        got = coll.get(include=["documents", "metadatas"], limit=min(limit, 100))
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        sample = []
        for i in range(len(ids)):
            meta = (metas[i] if i < len(metas) else {}) or {}
            content = (docs[i] if i < len(docs) else "") or ""
            sample.append({
                "id": ids[i],
                "doc_id": meta.get("doc_id"),
                "page_refs": meta.get("page_refs"),
                "page": meta.get("page"),
                "content_preview": content[:300].replace("\n", " ") + ("..." if len(content) > 300 else ""),
            })
        return {"available": True, "source": "chroma", "collection": coll.name, "total_chunks": count, "sample": sample}
    return _vector_store_from_sqlite(base, limit)


@app.get("/db/vector/search")
def db_vector_search(q: str = "", limit: int = 5):
    """Semantic search on vector store for debugging. Use ?q=revenue&limit=5 (requires Chroma API)."""
    base = _base()
    vs = VectorStore(persist_directory=base / "vector_store")
    if vs._collection is None:
        return {
            "available": False,
            "hits": [],
            "message": "Chroma API not available. Install chromadb for semantic search. Use /db/vector to view stored chunks.",
        }
    if not q.strip():
        return {"available": True, "hits": [], "message": "Use ?q=your+query"}
    hits = vs.search(q.strip(), k=min(limit, 20))
    out = []
    for ldu, score, doc_id in hits:
        page = ldu.page_refs[0] if ldu.page_refs else (ldu.bbox.page if ldu.bbox else None)
        out.append({
            "doc_id": doc_id,
            "page": page,
            "score": round(score, 4),
            "content_preview": ldu.content[:400].replace("\n", " "),
        })
    return {"available": True, "query": q, "hits": out}


@app.get("/db/facts")
def db_facts_summary(limit: int = 20):
    """View facts DB: documents list, fact count, sample facts."""
    base = _base()
    path = base / "facts.db"
    if not path.exists():
        return {"available": False, "documents": [], "facts_count": 0, "sample": []}
    store = FactStore(db_path=path)
    docs = store.query("SELECT doc_id FROM documents ORDER BY doc_id")
    facts_count = store.query("SELECT COUNT(*) AS n FROM facts")[0]["n"]
    sample = store.query("SELECT doc_id, page, fact_key, fact_value FROM facts ORDER BY id LIMIT ?", (min(limit, 100),))
    return {
        "available": True,
        "documents": [r["doc_id"] for r in docs],
        "facts_count": facts_count,
        "sample": sample,
    }


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


class AuditBody(BaseModel):
    claim: str
    doc_ids: list[str] | None = None


@app.post("/audit/verify")
def audit_verify(body: AuditBody):
    """Audit mode: verify a claim against the corpus. Returns verified (with ProvenanceChain) or unverifiable."""
    base = _base()
    profiles_dir = base / "profiles"
    doc_id_to_name = {p.stem: p.stem for p in profiles_dir.glob("*.json")} if profiles_dir.exists() else {}
    try:
        from src.agents import QueryAgent, Verified, verify_claim
        vs = VectorStore(persist_directory=base / "vector_store")
        store = FactStore(db_path=base / "facts.db")
        agent = QueryAgent(
            pageindex_dir=base / "pageindex",
            vector_store=vs,
            fact_store=store,
            doc_id_to_name=doc_id_to_name,
        )
        result = verify_claim(body.claim, agent, doc_ids=body.doc_ids or None, k=5, min_citations=1)
        if isinstance(result, Verified):
            return {
                "status": "verified",
                "provenance": [c.model_dump() for c in result.provenance.citations],
            }
        return {"status": "unverifiable", "reason": result.reason}
    except ApiKeyRequiredError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query")
def query(body: QueryBody):
    """Return answer and ProvenanceChain (citations with document_name, page_number, bbox, content_hash)."""
    base = _base()
    doc_ids = body.doc_ids if body.doc_ids else None
    profiles_dir = base / "profiles"
    all_profiles = list(profiles_dir.glob("*.json")) if profiles_dir.exists() else []
    doc_id_to_name = {p.stem: p.stem for p in all_profiles}
    if not doc_ids and len(all_profiles) == 1:
        doc_ids = [all_profiles[0].stem]
    try:
        from src.agents import QueryAgent
        vs = VectorStore(persist_directory=base / "vector_store")
        store = FactStore(db_path=base / "facts.db")
        agent = QueryAgent(
            pageindex_dir=base / "pageindex",
            vector_store=vs,
            fact_store=store,
            doc_id_to_name=doc_id_to_name,
        )
        answer, chain = agent.query(body.question, doc_ids=doc_ids, k=10)
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
    #uploadStatus, #queryResult, #queryProvenance, #auditResult { margin-top: 0.5rem; padding: 0.75rem; background: #f5f5f5; border-radius: 4px; font-size: 0.9rem; white-space: pre-wrap; }
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
      <span style="font-size:0.85rem;color:#666"> Select one or more documents to search; if none selected, searches all (or the only document if there is just one).</span>
    </p>
    <p>
      <input type="text" id="question" placeholder="Your question..." style="width:100%;max-width:400px;padding:0.5rem;">
      <button id="queryBtn">Ask</button>
    </p>
    <div id="queryResult"></div>
    <div id="queryProvenance" class="citations"></div>
  </section>

  <section>
    <h2>3. Audit mode — verify a claim</h2>
    <p>Enter a claim (e.g. &quot;The report states revenue was $4.2B in Q3&quot;). The system returns <strong>verified</strong> with source citations or <strong>unverifiable</strong>.</p>
    <p>
      <input type="text" id="claim" placeholder="Claim to verify..." style="width:100%;max-width:400px;padding:0.5rem;">
      <button id="auditBtn">Verify</button>
    </p>
    <div id="auditResult"></div>
  </section>

  <section>
    <h2>4. View databases (debug)</h2>
    <p>Open these in the browser to inspect what is stored (for debugging citations):</p>
    <ul style="font-size:0.9rem;">
      <li><a href="/db/vector" target="_blank">/db/vector</a> — Vector store summary and sample chunks</li>
      <li><a href="/db/vector/search?q=revenue&limit=5" target="_blank">/db/vector/search?q=revenue&limit=5</a> — Run a semantic search</li>
      <li><a href="/db/facts" target="_blank">/db/facts</a> — Facts DB (documents + sample facts)</li>
    </ul>
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
        setTimeout(function() {
          const sel = el("docIds");
          const opt = Array.from(sel.options).find(function(o) { return o.value === d.doc_id; });
          if (opt) { opt.selected = true; }
        }, 100);
      } catch (e) {
        show("uploadStatus", "Error: " + e.message, true);
      }
      el("uploadBtn").disabled = false;
    };

    el("queryBtn").onclick = async () => {
      const question = el("question").value.trim();
      if (!question) { show("queryResult", "Enter a question.", true); return; }
      const sel = el("docIds");
      let docIds = Array.from(sel.selectedOptions).map(o => o.value);
      const allDocIds = Array.from(sel.options).map(o => o.value);
      if (allDocIds.length === 0) {
        show("queryResult", "No processed documents yet. Upload and process at least one PDF in step 1, then try again.", true);
        return;
      }
      if (docIds.length === 0) docIds = allDocIds;
      show("queryResult", "Searching your documents...");
      el("queryResult").className = "";
      el("queryProvenance").textContent = "";
      el("queryBtn").disabled = true;
      try {
        const r = await api("/query", { method: "POST", body: JSON.stringify({ question, doc_ids: docIds }) });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || r.statusText);
        show("queryResult", d.answer || "No answer.");
        el("queryProvenance").innerHTML = (d.provenance || []).length
          ? "ProvenanceChain: " + (d.provenance || []).map(c => c.document_name + " p." + c.page_number + (c.content_hash ? " hash=" + c.content_hash.slice(0,8) : "")).join("; ")
          : "";
      } catch (e) {
        show("queryResult", "Error: " + e.message, true);
      }
      el("queryBtn").disabled = false;
    };

    el("auditBtn").onclick = async () => {
      const claim = el("claim").value.trim();
      if (!claim) { show("auditResult", "Enter a claim to verify.", true); return; }
      const sel = el("docIds");
      let docIds = Array.from(sel.selectedOptions).map(o => o.value);
      const allDocIds = Array.from(sel.options).map(o => o.value);
      if (docIds.length === 0) docIds = allDocIds;
      show("auditResult", "Verifying...");
      el("auditResult").className = "";
      el("auditBtn").disabled = true;
      try {
        const r = await api("/audit/verify", { method: "POST", body: JSON.stringify({ claim, doc_ids: docIds.length ? docIds : null }) });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || r.statusText);
        if (d.status === "verified") {
          show("auditResult", "Verified. Sources: " + (d.provenance || []).map(c => c.document_name + " p." + c.page_number).join("; "));
        } else {
          show("auditResult", "Unverifiable: " + (d.reason || "No supporting source found."), true);
        }
      } catch (e) {
        show("auditResult", "Error: " + e.message, true);
      }
      el("auditBtn").disabled = false;
    };
  </script>
</body>
</html>
"""
