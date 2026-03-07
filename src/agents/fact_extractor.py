from pathlib import Path

from src.models import DocumentProfile, ExtractedDocument, LDU
from src.utils.fact_store import FactStore, extract_facts


def extract_fact_table(
    doc: ExtractedDocument,
    ldus: list[LDU],
    profile: DocumentProfile,
) -> list[dict]:
    return extract_facts(doc, ldus=ldus, domain_hint=profile.domain_hint)


def extract_and_store_facts(
    doc_id: str,
    doc: ExtractedDocument,
    ldus: list[LDU],
    profile: DocumentProfile,
    store: FactStore,
    page: int = 0,
) -> int:
    facts = extract_fact_table(doc, ldus, profile)
    if not facts:
        return 0
    store.add_document_facts(doc_id, facts, page=page)
    return len(facts)
