from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.models import LDU


def _parse_metadata_json(s: Any) -> dict:
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {}

try:
    import chromadb
    from chromadb.config import Settings

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class VectorStore:
    def __init__(self, persist_directory: Path | str | None = None, collection_name: str = "refinery_ldus"):
        self._persist = str(persist_directory) if persist_directory else None
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._fallback: list[tuple[str, str, dict[str, Any]]] = []
        if _CHROMA_AVAILABLE and self._persist:
            self._client = chromadb.PersistentClient(path=self._persist, settings=Settings(anonymized_telemetry=False))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "LDU chunks for semantic search"},
            )
        elif _CHROMA_AVAILABLE:
            self._client = chromadb.Client(Settings(anonymized_telemetry=False))
            self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_document(self, doc_id: str, ldus: list[LDU]) -> None:
        if not ldus:
            return
        ids = []
        documents = []
        metadatas = []
        for i, ldu in enumerate(ldus):
            uid = f"{doc_id}__{i}"
            ids.append(uid)
            documents.append(ldu.content)
            meta: dict[str, Any] = {
                "doc_id": doc_id,
                "page_refs": ",".join(map(str, ldu.page_refs)),
                "content_hash": ldu.content_hash or "",
                "chunk_type": ldu.chunk_type.value if hasattr(ldu.chunk_type, "value") else str(ldu.chunk_type),
                "parent_section": ldu.parent_section or "",
            }
            if ldu.bbox:
                meta["page"] = ldu.bbox.page
                meta["x0"] = ldu.bbox.x0
                meta["y0"] = ldu.bbox.y0
                meta["x1"] = ldu.bbox.x1
                meta["y1"] = ldu.bbox.y1
            if ldu.metadata:
                meta["metadata_json"] = json.dumps(ldu.metadata)
            metadatas.append(meta)
        if self._collection is not None:
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        else:
            for uid, doc, meta in zip(ids, documents, metadatas):
                self._fallback.append((uid, doc, meta))

    def _search_via_sqlite(
        self, query: str, k: int, doc_ids: list[str] | None
    ) -> list[tuple[LDU, float, str]]:
        """Keyword search over Chroma SQLite when Chroma API is unavailable."""
        if not self._persist:
            return []
        db_path = Path(self._persist) / "chroma.sqlite3"
        if not db_path.exists():
            return []
        from src.models import BoundingBox, ChunkType
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("""
                SELECT e.embedding_id,
                       max(CASE WHEN m.key = 'doc_id' THEN m.string_value END) AS doc_id,
                       max(CASE WHEN m.key = 'page_refs' THEN m.string_value END) AS page_refs,
                       max(CASE WHEN m.key = 'page' THEN m.int_value END) AS page,
                       max(CASE WHEN m.key = 'x0' THEN m.float_value END) AS x0,
                       max(CASE WHEN m.key = 'y0' THEN m.float_value END) AS y0,
                       max(CASE WHEN m.key = 'x1' THEN m.float_value END) AS x1,
                       max(CASE WHEN m.key = 'y1' THEN m.float_value END) AS y1,
                       max(CASE WHEN m.key = 'parent_section' THEN m.string_value END) AS parent_section,
                       max(CASE WHEN m.key = 'content_hash' THEN m.string_value END) AS content_hash,
                       max(CASE WHEN m.key = 'chunk_type' THEN m.string_value END) AS chunk_type,
                       max(CASE WHEN m.key = 'chroma:document' THEN m.string_value END) AS content
                FROM embeddings e
                JOIN embedding_metadata m ON e.id = m.id
                GROUP BY e.id
            """)
            rows = cur.fetchall()
        finally:
            conn.close()
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 1]
        scored = []
        for r in rows:
            doc_id = r["doc_id"] or ""
            if doc_ids and doc_id not in doc_ids:
                continue
            content = (r["content"] or "") or ""
            score = (
                sum(1 for w in query_words if w in content.lower()) / max(len(query_words), 1)
                if query_words else 0.5
            )
            scored.append((score, r, content))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, r, content in scored[:k]:
            doc_id = r["doc_id"] or ""
            pr = (r["page_refs"] or "").strip()
            page_refs = [int(x) for x in pr.split(",") if x.strip()]
            if not page_refs and r["page"] is not None:
                page_refs = [int(r["page"])]
            bbox = None
            if r["page"] is not None:
                bbox = BoundingBox(
                    x0=float(r["x0"] or 0),
                    y0=float(r["y0"] or 0),
                    x1=float(r["x1"] or 0),
                    y1=float(r["y1"] or 0),
                    page=int(r["page"]),
                )
            ct = (r["chunk_type"] or "text").strip()
            try:
                ct = ChunkType(ct)
            except ValueError:
                ct = ChunkType.TEXT
            ldu = LDU(
                content=content,
                chunk_type=ct,
                page_refs=page_refs,
                bbox=bbox,
                parent_section=(r["parent_section"] or "") or "",
                token_count=0,
                content_hash=(r["content_hash"] or "") or "",
                metadata={},
            )
            out.append((ldu, min(1.0, score), doc_id))
        return out

    def _search_by_get_and_keyword(
        self, query: str, k: int, doc_ids: list[str]
    ) -> list[tuple[LDU, float, str]]:
        if not self._collection or not doc_ids:
            return []
        try:
            got = self._collection.get(
                where={"doc_id": {"$eq": doc_ids[0]}},
                include=["documents", "metadatas"],
                limit=300,
            )
        except Exception:
            return []
        if not got or not got.get("ids"):
            return []
        ids_list = got["ids"]
        docs_list = got.get("documents") or []
        metas_list = got.get("metadatas") or []
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 1]
        scored = []
        for i in range(len(ids_list)):
            doc_content = (docs_list[i] if i < len(docs_list) else "") or ""
            meta = (metas_list[i] if i < len(metas_list) else {}) or {}
            doc_id = meta.get("doc_id", "")
            score = (
                sum(1 for w in query_words if w in doc_content.lower()) / max(len(query_words), 1)
                if query_words
                else 0.5
            )
            scored.append((score, doc_content, meta, doc_id))
        scored.sort(key=lambda x: -x[0])
        from src.models import BoundingBox, ChunkType

        out = []
        for score, doc_content, meta, doc_id in scored[:k]:
            page_refs = [int(x) for x in (meta.get("page_refs") or "").split(",") if x.strip()]
            bbox = None
            if meta.get("page") is not None:
                bbox = BoundingBox(
                    x0=float(meta.get("x0", 0)),
                    y0=float(meta.get("y0", 0)),
                    x1=float(meta.get("x1", 0)),
                    y1=float(meta.get("y1", 0)),
                    page=int(meta.get("page", 0)),
                )
            ct = meta.get("chunk_type", "text")
            try:
                ct = ChunkType(ct)
            except ValueError:
                ct = ChunkType.TEXT
            ldu = LDU(
                content=doc_content,
                chunk_type=ct,
                page_refs=page_refs,
                bbox=bbox,
                parent_section=meta.get("parent_section") or "",
                token_count=0,
                content_hash=meta.get("content_hash") or "",
                metadata=_parse_metadata_json(meta.get("metadata_json")),
            )
            out.append((ldu, min(1.0, score), doc_id))
        return out

    def search(
        self,
        query: str,
        k: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[LDU, float, str]]:
        if self._collection is not None:
            where = None
            if doc_ids:
                where = {"doc_id": {"$in": list(doc_ids)}} if len(doc_ids) > 1 else {"doc_id": {"$eq": doc_ids[0]}}
            res = self._collection.query(
                query_texts=[query],
                n_results=min(k, 100),
                where=where,
                include=["documents", "distances", "metadatas"],
            )
            if not res or not res["ids"] or not res["ids"][0]:
                if doc_ids:
                    res = self._collection.query(
                        query_texts=[query],
                        n_results=200,
                        include=["documents", "distances", "metadatas"],
                    )
                    if res and res.get("ids") and res["ids"][0] and res.get("metadatas"):
                        doc_ids_set = set(doc_ids)
                        keep = [i for i, m in enumerate(res["metadatas"][0]) if (m or {}).get("doc_id") in doc_ids_set]
                        if keep:
                            res = {
                                "ids": [[res["ids"][0][j] for j in keep]],
                                "documents": [[(res["documents"] or [[]])[0][j] for j in keep]],
                                "metadatas": [[res["metadatas"][0][j] for j in keep]],
                                "distances": [[(res["distances"] or [[]])[0][j] for j in keep]] if res.get("distances") else [],
                            }
                        else:
                            res = {}
                    if not res or not res.get("ids") or not res["ids"][0]:
                        out = self._search_by_get_and_keyword(query, k, doc_ids)
                        if out:
                            return out
                if not res or not res.get("ids") or not res["ids"][0]:
                    return []
            ids = res["ids"][0]
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
            out = []
            for i in range(len(ids)):
                meta = (metas or [{}])[i] if metas else {}
                doc_id = meta.get("doc_id", "")
                doc_content = (docs[i] if docs and i < len(docs) else "") or ""
                page_refs = [int(x) for x in (meta.get("page_refs") or "").split(",") if x.strip()]
                bbox = None
                if meta.get("page") is not None:
                    from src.models import BoundingBox

                    bbox = BoundingBox(
                        x0=float(meta.get("x0", 0)),
                        y0=float(meta.get("y0", 0)),
                        x1=float(meta.get("x1", 0)),
                        y1=float(meta.get("y1", 0)),
                        page=int(meta.get("page", 0)),
                    )
                from src.models import ChunkType

                ct = meta.get("chunk_type", "text")
                try:
                    ct = ChunkType(ct)
                except ValueError:
                    ct = ChunkType.TEXT
                ldu = LDU(
                    content=doc_content,
                    chunk_type=ct,
                    page_refs=page_refs,
                    bbox=bbox,
                    parent_section=meta.get("parent_section") or "",
                    token_count=0,
                    content_hash=meta.get("content_hash") or "",
                    metadata=_parse_metadata_json(meta.get("metadata_json")),
                )
                dist = dists[i] if i < len(dists) else 0.0
                score = 1.0 / (1.0 + float(dist)) if dist else 1.0
                out.append((ldu, score, doc_id))
            return out
        hits = self._search_via_sqlite(query, k, doc_ids)
        if hits:
            return hits
        query_lower = query.lower()
        scored = []
        for uid, doc, meta in self._fallback:
            doc_id = meta.get("doc_id", "")
            if doc_ids and doc_id not in doc_ids:
                continue
            score = sum(1 for w in query_lower.split() if len(w) > 2 and w in doc.lower()) / max(len(query_lower.split()), 1)
            scored.append((score, uid, doc, meta))
        scored.sort(key=lambda x: -x[0])
        from src.models import BoundingBox, ChunkType

        out = []
        for score, uid, doc, meta in scored[:k]:
            doc_id = meta.get("doc_id", "")
            page_refs = [int(x) for x in (meta.get("page_refs") or "").split(",") if x.strip()]
            bbox = None
            if meta.get("page") is not None:
                bbox = BoundingBox(
                    x0=float(meta.get("x0", 0)),
                    y0=float(meta.get("y0", 0)),
                    x1=float(meta.get("x1", 0)),
                    y1=float(meta.get("y1", 0)),
                    page=int(meta.get("page", 0)),
                )
            ct = meta.get("chunk_type", "text")
            try:
                ct = ChunkType(ct)
            except ValueError:
                ct = ChunkType.TEXT
            ldu = LDU(
                content=doc,
                chunk_type=ct,
                page_refs=page_refs,
                bbox=bbox,
                parent_section=meta.get("parent_section") or "",
                token_count=0,
                content_hash=meta.get("content_hash") or "",
                metadata=_parse_metadata_json(meta.get("metadata_json")),
            )
            out.append((ldu, min(1.0, score), doc_id))
        return out
