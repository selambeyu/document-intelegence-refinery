from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Callable

from src.models import (
    ChunkType,
    ExtractedDocument,
    LDU,
    PageIndex,
    PageIndexSection,
)


def _heading_level(title: str) -> int:
    s = title.strip()
    if not s:
        return 0
    m = re.match(r"^(\d+(?:\.\d+)*)[.\s]", s)
    if m:
        return len(m.group(1).split("."))
    if re.match(r"^(?:Chapter|Part|Section)\s+\d+", s, re.I):
        return 1
    return 1


def _extract_key_entities(text: str, max_entities: int = 8) -> list[str]:
    entities: list[str] = []
    text = text.replace("\n", " ")
    for m in re.finditer(r"\b(?:FY|F\.Y\.?|fiscal year)\s*[\s\-/]?\s*(\d{4}(?:/\d{2})?)\b", text, re.I):
        entities.append(f"FY {m.group(1).strip()}")
    for m in re.finditer(r"\b(\d{1,2})[\s\-/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-/]\d{2,4}\b", text, re.I):
        entities.append(m.group(0))
    for m in re.finditer(r"\b(?:[\d,]+(?:\.\d+)?)\s*(?:percent|%|million|billion|Birr|ETB|USD)\b", text, re.I):
        entities.append(m.group(0).strip())
    for m in re.finditer(r"\b(?:Ministry of|Federal|National|Ethiopian)\s+[\w\s]{2,40}\b", text):
        ent = m.group(0).strip()
        if len(ent) <= 50 and ent not in entities:
            entities.append(ent)
    for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text):
        ent = m.group(0).strip()
        if 3 <= len(ent) <= 45 and ent not in entities and ent.lower() not in ("the", "and", "for", "with"):
            entities.append(ent)
    seen = set()
    out = []
    for e in entities:
        k = e.lower()[:40]
        if k not in seen and len(out) < max_entities:
            seen.add(k)
            out.append(e)
    return out[:max_entities]


def _default_summary(section_title: str, section_text: str, max_chars: int = 200) -> str:
    if not section_text.strip():
        return section_title or ""
    return (section_text.strip()[:max_chars] + "…") if len(section_text) > max_chars else section_text.strip()


def _summarize_via_llm(section_title: str, section_text: str, api_key: str | None = None) -> str:
    from src.utils.llm import use_ollama
    if use_ollama():
        try:
            from src.utils.llm import create_llm
            from langchain_core.messages import HumanMessage
            llm = create_llm()
            prompt = f"Summarize in 2-3 sentences (for a document index). Section: {section_title}\n\nContent:\n{section_text[:3000]}"
            out = llm.invoke([HumanMessage(content=prompt)])
            content = getattr(out, "content", "") or ""
            return content.strip() if content else _default_summary(section_title, section_text)
        except Exception:
            return _default_summary(section_title, section_text)
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key or not section_text.strip():
        return _default_summary(section_title, section_text)
    prompt = f"Summarize in 2-3 sentences (for a document index). Section: {section_title}\n\nContent:\n{section_text[:3000]}"
    url = "https://openrouter.ai/api/v1/chat/completions"
    body = {
        "model": "google/gemini-flash-1.5",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode())
        content = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return content.strip() if content else _default_summary(section_title, section_text)
    except Exception:
        return _default_summary(section_title, section_text)


class PageIndexBuilder:
    def __init__(
        self,
        summary_fn: Callable[[str, str], str] | None = None,
        use_llm_summary: bool = False,
    ):
        self._summary_fn = summary_fn or (_summarize_via_llm if use_llm_summary else _default_summary)

    def build(self, doc: ExtractedDocument, ldus: list[LDU]) -> PageIndex:
        by_section: dict[str, list[LDU]] = {}
        for ldu in ldus:
            sec = ldu.parent_section.strip() or "(no section)"
            by_section.setdefault(sec, []).append(ldu)

        flat: list[PageIndexSection] = []
        for title, chunk_ldus in by_section.items():
            if title == "(no section)" and not chunk_ldus:
                continue
            page_refs = []
            data_types: set[str] = set()
            parts: list[str] = []
            for ldu in chunk_ldus:
                page_refs.extend(ldu.page_refs)
                if ldu.chunk_type == ChunkType.TABLE:
                    data_types.add("tables")
                elif ldu.chunk_type == ChunkType.FIGURE:
                    data_types.add("figures")
                elif ldu.content.strip():
                    parts.append(ldu.content.strip())
            page_start = min(page_refs) if page_refs else 0
            page_end = max(page_refs) if page_refs else 0
            section_text = "\n\n".join(parts)[:5000]
            summary = self._summary_fn(title, section_text)
            key_entities = _extract_key_entities(section_text)
            flat.append(
                PageIndexSection(
                    title=title,
                    page_start=page_start,
                    page_end=page_end,
                    child_sections=[],
                    key_entities=key_entities,
                    summary=summary,
                    data_types_present=sorted(data_types),
                )
            )

        flat.sort(key=lambda s: (s.page_start, s.page_end))
        root = self._build_hierarchy(flat)
        return PageIndex(sections=root)

    def _build_hierarchy(self, flat: list[PageIndexSection]) -> list[PageIndexSection]:
        if not flat:
            return []
        levels = [_heading_level(s.title) for s in flat]
        stack: list[tuple[int, PageIndexSection]] = []
        root: list[PageIndexSection] = []

        for i, sec in enumerate(flat):
            level = levels[i]
            node = PageIndexSection(
                title=sec.title,
                page_start=sec.page_start,
                page_end=sec.page_end,
                child_sections=[],
                key_entities=sec.key_entities,
                summary=sec.summary,
                data_types_present=sec.data_types_present,
            )
            while stack and stack[-1][0] >= level:
                stack.pop()
            if not stack:
                root.append(node)
            else:
                parent = stack[-1][1]
                parent.child_sections.append(node)
            stack.append((level, node))

        return root


def build_pageindex(
    doc: ExtractedDocument,
    ldus: list[LDU],
    summary_fn: Callable[[str, str], str] | None = None,
    use_llm_summary: bool = False,
) -> PageIndex:
    builder = PageIndexBuilder(summary_fn=summary_fn, use_llm_summary=use_llm_summary)
    return builder.build(doc, ldus)


def save_pageindex(index: PageIndex, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.model_dump_json(indent=2), encoding="utf-8")


def load_pageindex(path: Path | str) -> PageIndex:
    return PageIndex.model_validate_json(Path(path).read_text(encoding="utf-8"))


def pageindex_search(index: PageIndex, topic: str, top_k: int = 3) -> list[PageIndexSection]:
    def collect(sec: PageIndexSection) -> list[PageIndexSection]:
        out = [sec]
        for ch in sec.child_sections:
            out.extend(collect(ch))
        return out

    all_secs = []
    for s in index.sections:
        all_secs.extend(collect(s))
    topic_lower = topic.lower()
    scored = []
    for sec in all_secs:
        score = 0.0
        if topic_lower in sec.title.lower():
            score += 2
        if topic_lower in sec.summary.lower():
            score += 1
        for ent in sec.key_entities:
            if topic_lower in ent.lower():
                score += 1.5
        for w in topic_lower.split():
            if len(w) > 2 and w in sec.title.lower():
                score += 1
            if len(w) > 2 and w in sec.summary.lower():
                score += 0.5
            for ent in sec.key_entities:
                if len(w) > 2 and w in ent.lower():
                    score += 0.5
        scored.append((score, sec))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_k]]
