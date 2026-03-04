from enum import Enum
from pydantic import BaseModel, Field


class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"


class ExtractionCost(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"


class DomainHint(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class DetectedLanguage(BaseModel):
    code: str = "und"
    confidence: float = 0.0


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


class DocumentProfile(BaseModel):
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: DetectedLanguage = Field(default_factory=DetectedLanguage)
    domain_hint: DomainHint = DomainHint.GENERAL
    extraction_cost: ExtractionCost


class ExtractedBlock(BaseModel):
    text: str
    bbox: BoundingBox
    block_type: BlockType = BlockType.TEXT
    table_ref: int | None = None
    figure_ref: int | None = None


class ExtractedTable(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    page: int = 0


class ExtractedFigure(BaseModel):
    caption: str = ""
    bbox: BoundingBox | None = None
    page: int = 0


class ExtractedDocument(BaseModel):
    blocks: list[ExtractedBlock] = Field(default_factory=list)
    reading_order: list[str] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    figures: list[ExtractedFigure] = Field(default_factory=list)
    confidence: float = 0.0
    strategy_used: str = ""
    cost_actual: float | None = None


class ProvenanceCitation(BaseModel):
    document_name: str = ""
    page_number: int = 0
    bbox: BoundingBox | None = None
    content_hash: str = ""


class ProvenanceChain(BaseModel):
    citations: list[ProvenanceCitation] = Field(default_factory=list)


class LDU(BaseModel):
    content: str = ""
    chunk_type: str = "text"
    page_refs: list[int] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    parent_section: str = ""
    token_count: int = 0
    content_hash: str = ""


class PageIndexSection(BaseModel):
    title: str = ""
    page_start: int = 0
    page_end: int = 0
    child_sections: list["PageIndexSection"] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    summary: str = ""
    data_types_present: list[str] = Field(default_factory=list)


class PageIndex(BaseModel):
    sections: list[PageIndexSection] = Field(default_factory=list)


PageIndexSection.model_rebuild()
