"""
Example: run query agent and save Q&A with ProvenanceChain.
Usage:
  uv run python scripts/example_qa.py .refinery "Your question" [doc_id1 doc_id2 ...]
"""
import json
import sys
from pathlib import Path

from src.agents import QueryAgent
from src.utils import get_refinery_base
from src.utils.fact_store import FactStore
from src.utils.vector_store import VectorStore


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: example_qa.py <refinery_base> <question> [doc_id1 ...]", file=sys.stderr)
        sys.exit(1)
    base_arg = sys.argv[1]
    question = sys.argv[2]
    doc_ids = sys.argv[3:] if len(sys.argv) > 3 else None
    base = get_refinery_base() if base_arg == ".refinery" else Path(base_arg).resolve()
    vs = VectorStore(persist_directory=base / "vector_store")
    store = FactStore(db_path=base / "facts.db")
    agent = QueryAgent(
        pageindex_dir=base / "pageindex",
        vector_store=vs,
        fact_store=store,
    )
    answer, chain = agent.query(question, doc_ids=doc_ids, k=10)
    print("Answer:", answer[:500], "..." if len(answer) > 500 else "")
    print("ProvenanceChain:", len(chain.citations), "citations")
    for c in chain.citations[:5]:
        print("  -", c.document_name, "p.", c.page_number, (c.content_hash[:12] + "...") if c.content_hash else "")
    out = {"question": question, "answer": answer, "provenance": chain.model_dump()}
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "logs" / "example_qa.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("Saved to", base / "logs" / "example_qa.json")


if __name__ == "__main__":
    main()
