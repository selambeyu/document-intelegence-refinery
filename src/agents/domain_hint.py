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


class KeywordDomainHintClassifier:
    def __init__(self, keyword_map: dict[DomainHint, list[str]] | None = None):
        self.keyword_map = keyword_map or KEYWORD_MAP

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
