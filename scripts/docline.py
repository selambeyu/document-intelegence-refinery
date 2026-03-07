import argparse
import json
import time
from pathlib import Path

from docling.document_converter import DocumentConverter
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str,
                        default=".refinery/phase0/docling")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    metrics = []

    pdfs = sorted(data_dir.rglob("*.pdf"))
    for pdf in tqdm(pdfs, desc="Docling convert"):
        t0 = time.time()
        row = {"document": pdf.name, "status": "ok", "error": None}
        try:
            result = converter.convert(str(pdf))
            md = result.document.export_to_markdown()
            elapsed = time.time() - t0

            md_path = out_dir / f"{pdf.stem}.md"
            md_path.write_text(md, encoding="utf-8")

            table_lines = sum(1 for line in md.splitlines()
                              if line.strip().startswith("|"))
            heading_lines = sum(1 for line in md.splitlines()
                                if line.strip().startswith("#"))

            row.update(
                {
                    "seconds": elapsed,
                    "markdown_chars": len(md),
                    "table_line_count": table_lines,
                    "heading_line_count": heading_lines,
                    "markdown_path": str(md_path),
                }
            )
        except Exception as e:
            row.update({"status": "error", "error": str(
                e), "seconds": time.time() - t0})
        metrics.append(row)

    metrics_path = out_dir / "docling_metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Saved: {metrics_path}")
    print(f"Saved markdown files in: {out_dir}")


if __name__ == "__main__":
    main()
