from __future__ import annotations

import re
from dataclasses import dataclass

from src.agents.query_agent import QueryAgent
from src.models import ProvenanceChain, ProvenanceCitation


@dataclass
class Verified:
    provenance: ProvenanceChain


@dataclass
class Unverifiable:
    reason: str


def _citation_is_linked(c: ProvenanceCitation) -> bool:
    return bool(c.document_name and (c.page_number > 0 or (c.content_hash and c.content_hash.strip())))


def verify_claim(
    claim: str,
    agent: QueryAgent,
    doc_ids: list[str] | None = None,
    k: int = 5,
    min_citations: int = 1,
) -> Verified | Unverifiable:
    """Verify or reject a claim using linked sources (ProvenanceChain: document_name, page, bbox, content_hash)."""
    answer_text, chain = agent.query(claim, doc_ids=doc_ids, k=k)
    linked = [c for c in chain.citations if _citation_is_linked(c)]
    if len(linked) < min_citations:
        return Unverifiable(reason="No supporting source found for the claim (no linked citations with document, page, or content_hash).")
    claim_words = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", claim.lower()))
    answer_lower = answer_text.lower()
    overlap = sum(1 for w in claim_words if w in answer_lower)
    if not claim_words or overlap >= max(1, len(claim_words) // 3):
        return Verified(provenance=ProvenanceChain(citations=linked))
    return Unverifiable(reason="Retrieved sources did not substantiate the claim (answer does not overlap sufficiently with claim terms).")
