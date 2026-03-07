from __future__ import annotations

from dataclasses import dataclass

from src.agents.query_agent import QueryAgent
from src.models import ProvenanceChain


@dataclass
class Verified:
    provenance: ProvenanceChain


@dataclass
class Unverifiable:
    reason: str


def verify_claim(
    claim: str,
    agent: QueryAgent,
    doc_ids: list[str] | None = None,
    k: int = 5,
    min_citations: int = 1,
) -> Verified | Unverifiable:
    answer_text, chain = agent.query(claim, doc_ids=doc_ids, k=k)
    if len(chain.citations) >= min_citations and chain.citations:
        return Verified(provenance=chain)
    return Unverifiable(reason="No supporting source found for the claim.")
