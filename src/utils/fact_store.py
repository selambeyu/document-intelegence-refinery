from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from src.models import DomainHint, ExtractedDocument, LDU


def _extract_facts_from_text(text: str) -> list[tuple[str, str, str]]:
    facts: list[tuple[str, str, str]] = []
    patterns = [
        (r"(?:revenue|total revenue|net revenue)\s*[:\-=]?\s*([\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|M|B)?)", "revenue"),
        (r"(?:expense|total expense|expenses)\s*[:\-=]?\s*([\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|M|B)?)", "expense"),
        (r"(?:profit|net profit)\s*[:\-=]?\s*([\$€£]?\s*[\d,]+(?:\.\d+)?)", "profit"),
        (r"(?:fiscal year|FY|year ended)\s*[:\-=]?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|[A-Za-z]+\s+\d{4})", "fiscal_year"),
        (r"(?:date|as at|as of)\s*[:\-=]?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4}|\d{4}[-/]\d{2}[-/]\d{2})", "date"),
        (r"(?:assets|total assets)\s*[:\-=]?\s*([\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|M|B)?)", "assets"),
        (r"(?:liabilities|total liabilities)\s*[:\-=]?\s*([\$€£]?\s*[\d,]+(?:\.\d+)?)", "liabilities"),
        (r"(?:equity|total equity)\s*[:\-=]?\s*([\$€£]?\s*[\d,]+(?:\.\d+)?)", "equity"),
    ]
    for pat, key in patterns:
        for m in re.finditer(pat, text, re.I):
            val = m.group(1).strip()
            if val:
                facts.append((key, val, "text"))
    return facts


def extract_facts(doc: ExtractedDocument, ldus: list[LDU] | None = None, domain_hint: DomainHint = DomainHint.GENERAL) -> list[dict[str, Any]]:
    if domain_hint != DomainHint.FINANCIAL and domain_hint != DomainHint.GENERAL:
        return []
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    texts = []
    for block in doc.blocks:
        if block.text and block.text.strip():
            texts.append(block.text)
    for table in doc.tables:
        header_str = " ".join(table.headers)
        texts.append(header_str)
        for row in table.rows:
            texts.append(" ".join(str(c) for c in row))
    if ldus:
        for ldu in ldus:
            if ldu.content.strip():
                texts.append(ldu.content)
    combined = "\n".join(texts)
    extracted = _extract_facts_from_text(combined)
    for key, val, source in extracted:
        if (key, val) in seen:
            continue
        seen.add((key, val))
        facts.append({"fact_key": key, "fact_value": val, "source": source})
    return facts


class FactStore:
    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    page INTEGER DEFAULT 0,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    value_type TEXT DEFAULT 'text',
                    content_hash TEXT,
                    bbox_json TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(fact_key)")
            conn.commit()

    def add_document_facts(
        self,
        doc_id: str,
        facts: list[dict[str, Any]],
        page: int = 0,
        content_hash: str = "",
        bbox_json: str = "",
    ) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute("INSERT OR IGNORE INTO documents (doc_id) VALUES (?)", (doc_id,))
            for f in facts:
                conn.execute(
                    "INSERT INTO facts (doc_id, page, fact_key, fact_value, value_type, content_hash, bbox_json) VALUES (?,?,?,?,?,?,?)",
                    (
                        doc_id,
                        page,
                        f.get("fact_key", ""),
                        f.get("fact_value", ""),
                        f.get("value_type", "text"),
                        content_hash,
                        bbox_json,
                    ),
                )
            conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def get_facts_by_key(self, fact_key: str, doc_id: str | None = None) -> list[dict[str, Any]]:
        if doc_id:
            return self.query("SELECT * FROM facts WHERE fact_key = ? AND doc_id = ?", (fact_key, doc_id))
        return self.query("SELECT * FROM facts WHERE fact_key = ?", (fact_key,))
