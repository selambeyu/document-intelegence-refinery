from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import LDU

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
            metadatas.append(meta)
        if self._collection is not None:
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        else:
            for uid, doc, meta in zip(ids, documents, metadatas):
                self._fallback.append((uid, doc, meta))

    def search(
        self,
        query: str,
        k: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[LDU, float, str]]:
        if self._collection is not None:
            where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
            res = self._collection.query(
                query_texts=[query],
                n_results=min(k, 100),
                where=where,
                include=["documents", "distances", "metadatas"],
            )
            if not res or not res["ids"] or not res["ids"][0]:
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
                )
                dist = dists[i] if i < len(dists) else 0.0
                score = 1.0 / (1.0 + float(dist)) if dist else 1.0
                out.append((ldu, score, doc_id))
            return out
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
            )
            out.append((ldu, min(1.0, score), doc_id))
        return out
