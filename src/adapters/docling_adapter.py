from __future__ import annotations

from src.models import (
    BoundingBox,
    BlockType,
    ExtractedBlock,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
)


def _bbox_from_prov(prov_item, page: int) -> BoundingBox | None:
    if not prov_item:
        return None
    bbox = getattr(prov_item, "bbox", None)
    if bbox is None:
        return BoundingBox(x0=0, y0=0, x1=0, y1=0, page=page)
    l = getattr(bbox, "l", getattr(bbox, "left", getattr(bbox, "x0", 0)))
    t = getattr(bbox, "t", getattr(bbox, "top", getattr(bbox, "y0", 0)))
    r = getattr(bbox, "r", getattr(bbox, "right", getattr(bbox, "x1", 0)))
    b = getattr(bbox, "b", getattr(bbox, "bottom", getattr(bbox, "y1", 0)))
    return BoundingBox(x0=float(l), y0=float(t), x1=float(r), y1=float(b), page=page)


def _page_from_prov(prov_item) -> int:
    if not prov_item:
        return 1
    p = getattr(prov_item, "page_no", getattr(prov_item, "page", 0))
    return int(p) + 1 if isinstance(p, int) and p >= 0 else 1


def docling_result_to_document(conversion_result) -> ExtractedDocument:
    doc = conversion_result.document
    blocks: list[ExtractedBlock] = []
    tables: list[ExtractedTable] = []
    figures: list[ExtractedFigure] = []
    reading_order: list[str] = []
    block_idx = 0

    iterate = getattr(doc, "iterate_items", None)
    if iterate is None:
        texts = getattr(doc, "texts", []) or []
        for item in texts:
            prov = getattr(item, "prov", None)
            prov0 = prov[0] if prov else None
            page = _page_from_prov(prov0)
            bbox = _bbox_from_prov(prov0, page)
            if bbox is None:
                bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0, page=page)
            text = getattr(item, "text", None) or str(getattr(item, "content", ""))
            blocks.append(ExtractedBlock(text=text.strip(), bbox=bbox, block_type=BlockType.TEXT))
            reading_order.append(str(block_idx))
            block_idx += 1
        for item in getattr(doc, "tables", []) or []:
            prov = getattr(item, "prov", None)
            prov0 = prov[0] if prov else None
            page = _page_from_prov(prov0)
            bbox = _bbox_from_prov(prov0, page)
            try:
                df = item.export_to_dataframe(doc=doc)
                headers = list(df.columns.astype(str)) if hasattr(df, "columns") else []
                rows = df.values.tolist() if hasattr(df, "values") else []
                rows = [[str(c) for c in row] for row in rows]
            except Exception:
                headers = []
                rows = []
            tables.append(ExtractedTable(headers=headers, rows=rows, bbox=bbox, page=page))
            blocks.append(
                ExtractedBlock(
                    text="",
                    bbox=bbox or BoundingBox(x0=0, y0=0, x1=0, y1=0, page=page),
                    block_type=BlockType.TABLE,
                    table_ref=len(tables) - 1,
                )
            )
            reading_order.append(str(block_idx))
            block_idx += 1
        for item in getattr(doc, "pictures", []) or []:
            prov = getattr(item, "prov", None)
            prov0 = prov[0] if prov else None
            page = _page_from_prov(prov0)
            bbox = _bbox_from_prov(prov0, page)
            caption = getattr(item, "caption", None) or getattr(item, "title", "") or ""
            figures.append(ExtractedFigure(caption=str(caption), bbox=bbox, page=page))
            blocks.append(
                ExtractedBlock(
                    text=caption,
                    bbox=bbox or BoundingBox(x0=0, y0=0, x1=0, y1=0, page=page),
                    block_type=BlockType.FIGURE,
                    figure_ref=len(figures) - 1,
                )
            )
            reading_order.append(str(block_idx))
            block_idx += 1
        return ExtractedDocument(
            blocks=blocks,
            reading_order=reading_order,
            tables=tables,
            figures=figures,
            confidence=0.90,
            strategy_used="layout_docling",
        )

    for item, _level in iterate():
        prov = getattr(item, "prov", None)
        prov0 = prov[0] if prov else None
        page = _page_from_prov(prov0)
        bbox = _bbox_from_prov(prov0, page)
        if bbox is None:
            bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0, page=page)

        kind = type(item).__name__
        if "Table" in kind:
            try:
                df = item.export_to_dataframe(doc=doc)
                headers = list(df.columns.astype(str)) if hasattr(df, "columns") else []
                rows = [[str(c) for c in row] for row in (df.values.tolist() if hasattr(df, "values") else [])]
            except Exception:
                headers = []
                rows = []
            tables.append(ExtractedTable(headers=headers, rows=rows, bbox=bbox, page=page))
            blocks.append(
                ExtractedBlock(text="", bbox=bbox, block_type=BlockType.TABLE, table_ref=len(tables) - 1)
            )
        elif "Picture" in kind or "Image" in kind:
            caption = getattr(item, "caption", None) or getattr(item, "title", "") or ""
            figures.append(ExtractedFigure(caption=str(caption), bbox=bbox, page=page))
            blocks.append(
                ExtractedBlock(
                    text=caption,
                    bbox=bbox,
                    block_type=BlockType.FIGURE,
                    figure_ref=len(figures) - 1,
                )
            )
        else:
            text = getattr(item, "text", None) or str(getattr(item, "content", ""))
            blocks.append(ExtractedBlock(text=text.strip(), bbox=bbox, block_type=BlockType.TEXT))
        reading_order.append(str(block_idx))
        block_idx += 1

    return ExtractedDocument(
        blocks=blocks,
        reading_order=reading_order,
        tables=tables,
        figures=figures,
        confidence=0.90,
        strategy_used="layout_docling",
    )
