#!/usr/bin/env python3
"""
Inspect vector store (Chroma), Chroma SQLite, and facts SQLite to debug query/citation issues.

Usage (from repo root):
  python scripts/inspect_stores.py
  REFINERY_BASE=/path/to/.refinery python scripts/inspect_stores.py

What you see:
- Chroma (via API): collection name, chunk count, sample ids/metadata (doc_id, page_refs, page).
  If "collection not available", install chromadb and ensure REFINERY_BASE points to the same
  .refinery used when indexing.
- Chroma SQLite: table row counts and sample citation metadata (doc_id, page_refs, page) from
  embedding_metadata. Use this to verify what is stored for citations when the API is unavailable.
- Facts SQLite: documents + facts table row counts and sample rows.
- Sample search: one vector search and the resulting (doc_id, page, score) to debug missing citations.

Viewing stores manually:
  sqlite3 .refinery/vector_store/chroma.sqlite3   # Chroma metadata
  sqlite3 .refinery/facts.db                       # Extracted facts
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFINERY_BASE = Path(os.environ.get("REFINERY_BASE", "")).resolve() if os.environ.get("REFINERY_BASE") else None
BASE = (REFINERY_BASE if REFINERY_BASE else Path.cwd() / ".refinery").resolve()
VECTOR_DIR = BASE / "vector_store"
FACTS_DB = BASE / "facts.db"


def inspect_chroma_via_api():
    """Use VectorStore (Chroma client) to list collection and sample data."""
    from src.utils.vector_store import VectorStore

    if not VECTOR_DIR.exists():
        print("Vector store dir missing:", VECTOR_DIR)
        return
    vs = VectorStore(persist_directory=VECTOR_DIR)
    if vs._collection is None:
        print("Chroma collection not available (in-memory or chromadb not installed)")
        return
    coll = vs._collection
    count = coll.count()
    print("Chroma (via API)")
    print("  collection:", coll.name)
    print("  total chunks:", count)
    if count == 0:
        return
    got = coll.get(include=["documents", "metadatas"], limit=5)
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    print("  sample (first 5):")
    for i in range(min(5, len(ids))):
        meta = (metas[i] if i < len(metas) else {}) or {}
        doc_preview = (docs[i] if i < len(docs) else "")[:120].replace("\n", " ")
        print(f"    id={ids[i]!r} doc_id={meta.get('doc_id')!r} page_refs={meta.get('page_refs')!r} page={meta.get('page')}")
        print(f"      content: {doc_preview!r}...")


def inspect_chroma_sqlite():
    """Inspect Chroma's SQLite (metadata only; embeddings are in binary)."""
    chroma_sqlite = VECTOR_DIR / "chroma.sqlite3"
    if not chroma_sqlite.exists():
        print("Chroma SQLite not found:", chroma_sqlite)
        return
    print("\nChroma SQLite (chroma.sqlite3)")
    conn = sqlite3.connect(chroma_sqlite)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    print("  tables:", tables)
    for t in tables:
        cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        print(f"  {t}: {n} rows")
    if "embeddings" in tables:
        cur = conn.execute("SELECT * FROM embeddings LIMIT 2")
        cols = [d[0] for d in cur.description]
        print("  embeddings columns:", cols)
        for row in cur.fetchall():
            print("    row:", dict(zip(cols, row)))
    if "embedding_metadata" in tables:
        cur = conn.execute("""
            SELECT e.embedding_id,
                   max(CASE WHEN m.key = 'doc_id' THEN m.string_value END) AS doc_id,
                   max(CASE WHEN m.key = 'page_refs' THEN m.string_value END) AS page_refs,
                   max(CASE WHEN m.key = 'page' THEN m.int_value END) AS page
            FROM embeddings e
            JOIN embedding_metadata m ON e.id = m.id
            GROUP BY e.id
            LIMIT 8
        """)
        rows = cur.fetchall()
        print("  citation-related metadata (sample):")
        for r in rows:
            print("    ", r)
    conn.close()


def inspect_facts_db():
    """Inspect facts.db (documents + facts)."""
    if not FACTS_DB.exists():
        print("\nFacts DB not found:", FACTS_DB)
        return
    print("\nFacts SQLite (facts.db)")
    conn = sqlite3.connect(FACTS_DB)
    conn.row_factory = sqlite3.Row
    for name in ["documents", "facts"]:
        cur = conn.execute(f"SELECT COUNT(*) FROM {name}")
        print(f"  {name}: {cur.fetchone()[0]} rows")
    cur = conn.execute("SELECT * FROM facts LIMIT 5")
    rows = cur.fetchall()
    if rows:
        print("  sample facts:")
        for r in rows:
            print("   ", dict(r))
    conn.close()


def run_sample_search():
    """Run one semantic search and show what becomes citations."""
    from src.utils.vector_store import VectorStore

    if not VECTOR_DIR.exists():
        return
    vs = VectorStore(persist_directory=VECTOR_DIR)
    if vs._collection is None:
        return
    query = "revenue or financial"
    hits = vs.search(query, k=3)
    print("\nSample search (query=%r, k=3)" % query)
    print("  hits:", len(hits))
    for ldu, score, doc_id in hits:
        page = ldu.page_refs[0] if ldu.page_refs else (ldu.bbox.page if ldu.bbox else None)
        print(f"  doc_id={doc_id!r} page={page} score={score:.3f} content_hash={ldu.content_hash or ''!r}")
        print(f"    content: {ldu.content[:100]!r}...")
    if not hits:
        print("  (No hits — check if collection has documents and embedding model matches)")


def main():
    print("Refinery base:", BASE)
    print("vector_store:", VECTOR_DIR)
    print("facts.db:", FACTS_DB)
    inspect_chroma_via_api()
    inspect_chroma_sqlite()
    inspect_facts_db()
    run_sample_search()


if __name__ == "__main__":
    main()
