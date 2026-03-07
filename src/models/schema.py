from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


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


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"
    CAPTION = "caption"
    LIST = "list"


class DetectedLanguage(BaseModel):
    code: str = "und"
    confidence: float = 0.0


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    @model_validator(mode="before")
    @classmethod
    def bounds_order_before(cls, data):
        if not isinstance(data, dict):
            return data
        x0 = data.get("x0", 0)
        y0 = data.get("y0", 0)
        x1 = data.get("x1", 0)
        y1 = data.get("y1", 0)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return {**data, "x0": x0, "y0": y0, "x1": x1, "y1": y1}

    @model_validator(mode="after")
    def bounds_order_after(self):
        x0, x1 = min(self.x0, self.x1), max(self.x0, self.x1)
        y0, y1 = min(self.y0, self.y1), max(self.y0, self.y1)
        if (x0, x1, y0, y1) == (self.x0, self.x1, self.y0, self.y1):
            return self
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=self.page)


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
    review_flag: bool = False


class ProvenanceCitation(BaseModel):
    document_name: str = ""
    page_number: int = 0
    bbox: BoundingBox | None = None
    content_hash: str = ""

    @field_validator("content_hash")
    @classmethod
    def hash_non_empty_if_set(cls, v: str) -> str:
        if v is not None and len(v) > 0 and not v.strip():
            raise ValueError("content_hash must be non-empty when provided")
        return v or ""


class ProvenanceChain(BaseModel):
    citations: list[ProvenanceCitation] = Field(default_factory=list)

    @property
    def aggregate_bbox(self) -> BoundingBox | None:
        if not self.citations:
            return None
        boxes = [c.bbox for c in self.citations if c.bbox is not None]
        if not boxes:
            return None
        return BoundingBox(
            x0=min(b.x0 for b in boxes),
            y0=min(b.y0 for b in boxes),
            x1=max(b.x1 for b in boxes),
            y1=max(b.y1 for b in boxes),
            page=boxes[0].page,
        )

    @property
    def content_hashes(self) -> list[str]:
        return [c.content_hash for c in self.citations if c.content_hash]


class LDU(BaseModel):
    content: str = ""
    chunk_type: ChunkType = ChunkType.TEXT
    page_refs: list[int] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    parent_section: str = ""
    token_count: int = 0
    content_hash: str = ""

    @field_validator("content_hash")
    @classmethod
    def hash_non_empty_if_set(cls, v: str) -> str:
        if v is not None and len(v) > 0 and not v.strip():
            raise ValueError("content_hash must be non-empty when provided")
        return v or ""


class PageIndexSection(BaseModel):
    title: str = ""
    page_start: int = 0
    page_end: int = 0
    child_sections: list["PageIndexSection"] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    summary: str = ""
    data_types_present: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def page_end_not_before_start(self):
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class PageIndex(BaseModel):
    sections: list[PageIndexSection] = Field(default_factory=list)


PageIndexSection.model_rebuild()
