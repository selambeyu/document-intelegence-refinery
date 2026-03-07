from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from src.utils.llm import create_llm

from src.agents.indexer import load_pageindex, pageindex_search
from src.models import (
    BoundingBox,
    LDU,
    ProvenanceChain,
    ProvenanceCitation,
)
from src.utils.vector_store import VectorStore


def _ldu_to_citation(ldu: LDU, document_name: str) -> dict[str, Any]:
    page = ldu.page_refs[0] if ldu.page_refs else (ldu.bbox.page if ldu.bbox else 0)
    return {
        "document_name": document_name,
        "page_number": page,
        "bbox": ldu.bbox.model_dump() if ldu.bbox else None,
        "content_hash": ldu.content_hash or "",
    }


class QueryState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    citations: list[dict[str, Any]]
    doc_ids: list[str] | None


def pageindex_navigate(
    doc_id: str,
    topic: str,
    pageindex_dir: Path | str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    path = Path(pageindex_dir) / f"{doc_id}.json"
    if not path.exists():
        return []
    index = load_pageindex(path)
    sections = pageindex_search(index, topic, top_k=top_k)
    return [s.model_dump() for s in sections]


def semantic_search(
    query: str,
    vector_store: VectorStore,
    doc_ids: list[str] | None = None,
    k: int = 10,
) -> list[tuple[LDU, float, str]]:
    return vector_store.search(query, k=k, doc_ids=doc_ids)


def structured_query(
    sql: str,
    fact_store: Any,
    params: tuple = (),
) -> list[dict[str, Any]]:
    return fact_store.query(sql, params)


def _make_tools(
    pageindex_dir: Path,
    vector_store: VectorStore,
    fact_store: Any | None,
    doc_ids: list[str] | None,
    doc_id_to_name: dict[str, str],
) -> list:
    @tool
    def pageindex_navigate_tool(doc_id: str, topic: str, top_k: int = 3) -> str:
        """Navigate the document's PageIndex tree by topic. Use to find which sections cover a topic before searching. Returns section titles, page ranges, and summaries."""
        sections = pageindex_navigate(doc_id, topic, pageindex_dir, top_k=top_k)
        if not sections:
            return f"No sections found for doc_id={doc_id} topic={topic!r}."
        parts = [f"Section: {s.get('title', '')} (pp. {s.get('page_start', '')}-{s.get('page_end', '')}): {s.get('summary', '')[:200]}" for s in sections]
        return "\n".join(parts)

    @tool
    def semantic_search_tool(query: str, k: int = 10) -> str:
        """Semantic search over document chunks. Use for finding relevant text, tables, or figures. Returns chunk content and metadata (document, page)."""
        hits = semantic_search(query, vector_store, doc_ids=doc_ids, k=k)
        if not hits:
            return "No matching chunks found."
        out_lines = []
        for ldu, score, did in hits:
            doc_name = doc_id_to_name.get(did, did) or "document"
            out_lines.append(f"[{doc_name} p.{ldu.page_refs[0] if ldu.page_refs else '?'}] {ldu.content.strip()[:400]}")
        return "\n\n".join(out_lines[:15])

    @tool
    def structured_query_tool(sql: str) -> str:
        """Run a read-only SQL query over the extracted fact table (e.g. revenue, fiscal_year, assets). Table: facts(doc_id, page, fact_key, fact_value)."""
        if fact_store is None:
            return "Fact store not available."
        rows = structured_query(sql, fact_store)
        if not rows:
            return "No rows returned."
        return "\n".join(str(r) for r in rows[:50])

    return [pageindex_navigate_tool, semantic_search_tool, structured_query_tool]


def _agent_node(state: QueryState, llm_with_tools: Any) -> QueryState:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _tools_node(
    state: QueryState,
    pageindex_dir: Path,
    vector_store: VectorStore,
    fact_store: Any | None,
    doc_id_to_name: dict[str, str],
) -> QueryState:
    messages = state["messages"]
    citations = list(state.get("citations") or [])
    doc_ids = state.get("doc_ids")
    last = messages[-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {"messages": [], "citations": citations}
    tool_messages = []
    for tc in last.tool_calls:
        name = tc.get("name", "")
        args = tc.get("args") or {}
        tid = tc.get("id", "")
        if name == "pageindex_navigate_tool":
            sections = pageindex_navigate(
                args.get("doc_id", (doc_ids or [""])[0]),
                args.get("topic", ""),
                pageindex_dir,
                top_k=args.get("top_k", 3),
            )
            content = "\n".join(
                f"Section: {s.get('title', '')} (pp. {s.get('page_start')}-{s.get('page_end')}): {s.get('summary', '')[:200]}"
                for s in sections
            ) if sections else "No sections found."
        elif name == "semantic_search_tool":
            hits = semantic_search(
                args.get("query", ""),
                vector_store,
                doc_ids=doc_ids,
                k=args.get("k", 10),
            )
            for ldu, _, did in hits:
                doc_name = doc_id_to_name.get(did, did) or "document"
                citations.append(_ldu_to_citation(ldu, doc_name))
            content = "\n\n".join(
                f"[{doc_id_to_name.get(did, did) or did} p.{ldu.page_refs[0] if ldu.page_refs else '?'}] {ldu.content.strip()[:400]}"
                for ldu, _, did in hits[:15]
            ) if hits else "No matching chunks found."
        elif name == "structured_query_tool" and fact_store:
            rows = structured_query(args.get("sql", "SELECT 1"), fact_store)
            seen = set()
            for r in rows:
                if isinstance(r, dict):
                    did = r.get("doc_id")
                    page = r.get("page", 0)
                    if did and (did, page) not in seen:
                        seen.add((did, page))
                        doc_name = doc_id_to_name.get(did, did) or did
                        citations.append({
                            "document_name": doc_name,
                            "page_number": int(page) if page is not None else 0,
                            "bbox": None,
                            "content_hash": r.get("content_hash") or "",
                        })
            content = "\n".join(str(r) for r in rows[:50]) if rows else "No rows returned."
        else:
            content = "Tool not available or invalid arguments."
        tool_messages.append(ToolMessage(content=content, tool_call_id=tid))
    return {"messages": tool_messages, "citations": citations}


def _should_continue(state: QueryState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def build_query_graph(
    pageindex_dir: Path | str,
    vector_store: VectorStore,
    fact_store: Any | None = None,
    doc_id_to_name: dict[str, str] | None = None,
    llm: Any | None = None,
):
    pageindex_dir = Path(pageindex_dir)
    doc_id_to_name = doc_id_to_name or {}
    if llm is None:
        llm = create_llm()
    tools = _make_tools(pageindex_dir, vector_store, fact_store, None, doc_id_to_name)
    llm_with_tools = llm.bind_tools(tools)
    graph_builder = StateGraph(QueryState)
    graph_builder.add_node("agent", lambda s: _agent_node(s, llm_with_tools))
    graph_builder.add_node(
        "tools",
        lambda s: _tools_node(s, pageindex_dir, vector_store, fact_store, doc_id_to_name),
    )
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile()


class QueryAgent:
    def __init__(
        self,
        pageindex_dir: Path | str,
        vector_store: VectorStore,
        fact_store: Any | None = None,
        doc_id_to_name: dict[str, str] | None = None,
        llm: Any | None = None,
    ):
        self._pageindex_dir = Path(pageindex_dir)
        self._vector_store = vector_store
        self._fact_store = fact_store
        self._doc_names = doc_id_to_name or {}
        self._graph = build_query_graph(
            self._pageindex_dir,
            self._vector_store,
            self._fact_store,
            self._doc_names,
            llm=llm,
        )

    def pageindex_navigate(self, doc_id: str, topic: str, top_k: int = 3) -> list[dict]:
        return pageindex_navigate(doc_id, topic, self._pageindex_dir, top_k=top_k)

    def semantic_search(self, query: str, doc_ids: list[str] | None = None, k: int = 10) -> list[tuple[LDU, float, str]]:
        return semantic_search(query, self._vector_store, doc_ids=doc_ids, k=k)

    def structured_query(self, sql: str, params: tuple = ()) -> list[dict]:
        if self._fact_store is None:
            return []
        return structured_query(sql, self._fact_store, params)

    def query(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        k: int = 10,
    ) -> tuple[str, ProvenanceChain]:
        from src.utils.api_key import require_api_key_for_query
        from src.utils.llm import use_ollama
        if not use_ollama():
            require_api_key_for_query()
        doc_scope = ", ".join(doc_ids) if doc_ids else "all"
        pre_hits = semantic_search(question, self._vector_store, doc_ids=doc_ids, k=k)
        pre_citations: list[dict[str, Any]] = []
        retrieved_block = ""
        if pre_hits:
            for ldu, _, did in pre_hits[:10]:
                doc_name = self._doc_names.get(did, did) or did
                pre_citations.append(_ldu_to_citation(ldu, doc_name))
                retrieved_block += f"[{doc_name} p.{ldu.page_refs[0] if ldu.page_refs else '?'}]\n{ldu.content.strip()[:500]}\n\n"
        if not pre_hits and doc_ids:
            pre_hits = semantic_search(question, self._vector_store, doc_ids=None, k=k)
            if pre_hits:
                for ldu, _, did in pre_hits[:10]:
                    doc_name = self._doc_names.get(did, did) or did
                    pre_citations.append(_ldu_to_citation(ldu, doc_name))
                    retrieved_block += f"[{doc_name} p.{ldu.page_refs[0] if ldu.page_refs else '?'}]\n{ldu.content.strip()[:500]}\n\n"
                retrieved_block = "Note: No chunks matched the selected document(s); showing results from all documents.\n\n" + retrieved_block
        prompt_parts = [
            "Answer ONLY using the content returned by the tools or the retrieved content below. Do not use general knowledge.",
            "If you were given 'Retrieved content' below, use it to answer the question. Only say 'No relevant content found in the selected documents' if the tools or retrieved content explicitly say 'No matching chunks found' or 'No sections found' and there is no excerpt to use.",
            f"Search in docs: {doc_scope}.",
        ]
        if retrieved_block:
            prompt_parts.append("Retrieved content (use this to answer if it matches the question):\n" + retrieved_block.strip())
        prompt_parts.append(f"Question: {question}")
        initial: QueryState = {
            "messages": [HumanMessage(content="\n\n".join(prompt_parts))],
            "citations": pre_citations,
            "doc_ids": doc_ids,
        }
        final = self._graph.invoke(initial)
        messages = final.get("messages") or []
        citations_raw = final.get("citations") or pre_citations
        answer_text = "No answer produced."
        for m in reversed(messages):
            if hasattr(m, "content") and m.content and isinstance(m.content, str) and m.content.strip():
                answer_text = m.content.strip()
                break
        no_content_phrase = "no relevant content found in the selected documents"
        if citations_raw and no_content_phrase in answer_text.lower():
            excerpt = retrieved_block.strip()[:2000] if retrieved_block else ""
            if excerpt:
                answer_text = (
                    "The following excerpts from the selected documents may relate to your question:\n\n"
                    + excerpt
                    + "\n\n(Source pages and hashes are in the citations below. If this does not answer your question, try rephrasing or selecting different documents.)"
                )
        if doc_ids and not citations_raw and no_content_phrase not in answer_text.lower():
            answer_text = (
                "No relevant content found in the selected documents. "
                "The document may have no indexed text (e.g. scanned images without OCR) or processing may have produced no chunks—try re-processing with layout/vision extraction."
            )
        citations = [
            ProvenanceCitation(
                document_name=c.get("document_name", ""),
                page_number=c.get("page_number", 0),
                bbox=BoundingBox(**c["bbox"]) if isinstance(c.get("bbox"), dict) else c.get("bbox"),
                content_hash=c.get("content_hash", ""),
            )
            for c in citations_raw
        ]
        return answer_text, ProvenanceChain(citations=citations)
