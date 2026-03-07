from typing import Protocol

from src.models import DomainHint


class DomainHintClassifier(Protocol):
    def classify(self, text: str) -> DomainHint:
        ...


KEYWORD_MAP = {
    DomainHint.FINANCIAL: [
        "revenue", "expense", "balance sheet", "income statement", "fiscal", "audit",
        "profit", "loss", "assets", "liabilities", "equity", "cash flow", "annual report",
        "financial statements", "tax", "budget", "expenditure", "appropriation",
    ],
    DomainHint.LEGAL: [
        "whereas", "hereby", "hereinafter", "pursuant", "plaintiff", "defendant",
        "court", "agreement", "contract", "clause", "jurisdiction", "legal", "statute",
    ],
    DomainHint.TECHNICAL: [
        "implementation", "algorithm", "api", "software", "system", "protocol",
        "configuration", "architecture", "module", "framework", "database",
    ],
    DomainHint.MEDICAL: [
        "patient", "clinical", "diagnosis", "treatment", "medication", "symptom",
        "therapy", "health", "medical", "pharmaceutical", "dosage",
    ],
}


def _keyword_map_from_config(config: dict | None) -> dict[DomainHint, list[str]] | None:
    if not config or "domain_keywords" not in config:
        return None
    raw = config["domain_keywords"]
    if not raw or not isinstance(raw, dict):
        return None
    out: dict[DomainHint, list[str]] = {}
    for key, words in raw.items():
        if not isinstance(words, list):
            continue
        try:
            hint = DomainHint(key.lower()) if isinstance(key, str) else key
        except ValueError:
            continue
        if hint == DomainHint.GENERAL:
            continue
        out[hint] = [str(w) for w in words]
    return out if out else None


class KeywordDomainHintClassifier:
    def __init__(
        self,
        keyword_map: dict[DomainHint, list[str]] | None = None,
        config: dict | None = None,
    ):
        from_config = _keyword_map_from_config(config)
        self.keyword_map = keyword_map or from_config or KEYWORD_MAP

    def classify(self, text: str) -> DomainHint:
        if not text or not text.strip():
            return DomainHint.GENERAL
        combined = text.lower()
        best = DomainHint.GENERAL
        best_count = 0
        for domain, keywords in self.keyword_map.items():
            count = sum(1 for k in keywords if k in combined)
            if count > best_count:
                best_count = count
                best = domain
        return best
