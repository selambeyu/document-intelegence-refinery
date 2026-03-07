from typing import Protocol

from src.models import DomainHint


class DomainHintClassifier(Protocol):
    def classify(self, text: str) -> str:
        """Return best-matching domain key (e.g. 'financial', 'education'). Used for domain_id; enum mapping is done by caller."""
        ...


KEYWORD_MAP = {
    DomainHint.FINANCIAL.value: [
        "revenue", "expense", "balance sheet", "income statement", "fiscal", "audit",
        "profit", "loss", "assets", "liabilities", "equity", "cash flow", "annual report",
        "financial statements", "tax", "budget", "expenditure", "appropriation",
    ],
    DomainHint.LEGAL.value: [
        "whereas", "hereby", "hereinafter", "pursuant", "plaintiff", "defendant",
        "court", "agreement", "contract", "clause", "jurisdiction", "legal", "statute",
    ],
    DomainHint.TECHNICAL.value: [
        "implementation", "algorithm", "api", "software", "system", "protocol",
        "configuration", "architecture", "module", "framework", "database",
    ],
    DomainHint.MEDICAL.value: [
        "patient", "clinical", "diagnosis", "treatment", "medication", "symptom",
        "therapy", "health", "medical", "pharmaceutical", "dosage",
    ],
}


def _keyword_map_from_config(config: dict | None) -> dict[str, list[str]] | None:
    if not config or "domain_keywords" not in config:
        return None
    raw = config["domain_keywords"]
    if not raw or not isinstance(raw, dict):
        return None
    out: dict[str, list[str]] = {}
    for key, words in raw.items():
        if not isinstance(words, list):
            continue
        key_str = key.lower() if isinstance(key, str) else str(key)
        if key_str == DomainHint.GENERAL.value:
            continue
        out[key_str] = [str(w) for w in words]
    return out if out else None


class KeywordDomainHintClassifier:
    def __init__(
        self,
        keyword_map: dict[str, list[str]] | None = None,
        config: dict | None = None,
    ):
        from_config = _keyword_map_from_config(config)
        self.keyword_map = keyword_map or from_config or KEYWORD_MAP

    def classify(self, text: str) -> str:
        if not text or not text.strip():
            return DomainHint.GENERAL.value
        combined = text.lower()
        best = DomainHint.GENERAL.value
        best_count = 0
        for domain_key, keywords in self.keyword_map.items():
            count = sum(1 for k in keywords if k in combined)
            if count > best_count:
                best_count = count
                best = domain_key
        return best
