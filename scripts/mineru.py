import argparse
import json
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def run_mineru_on_pdf(pdf_path: Path, out_dir: Path) -> dict:
    doc_out = out_dir / pdf_path.stem
    doc_out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["mineru", "-p", str(pdf_path), "-o", str(doc_out)],
        check=True,
        capture_output=True,
        text=True,
    )
    md_path = doc_out / f"{pdf_path.stem}.md"
    if not md_path.exists():
        candidates = list(doc_out.glob("*.md"))
        md_path = candidates[0] if candidates else None
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Run MinerU on all PDFs in data dir")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default=".refinery/phase0/mineru")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(data_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {data_dir}")
        return

    metrics = []
    for pdf in tqdm(pdfs, desc="MinerU convert"):
        t0 = time.time()
        row = {"document": pdf.name, "status": "ok", "error": None}
        try:
            md_path = run_mineru_on_pdf(pdf, out_dir)
            elapsed = time.time() - t0
            if md_path and md_path.exists():
                md = md_path.read_text(encoding="utf-8")
                table_lines = sum(
                    1 for line in md.splitlines() if line.strip().startswith("|")
                )
                heading_lines = sum(
                    1 for line in md.splitlines() if line.strip().startswith("#")
                )
                row.update(
                    {
                        "seconds": elapsed,
                        "markdown_chars": len(md),
                        "table_line_count": table_lines,
                        "heading_line_count": heading_lines,
                        "markdown_path": str(md_path),
                    }
                )
            else:
                row.update({"status": "ok", "seconds": elapsed, "markdown_path": None})
        except subprocess.CalledProcessError as e:
            row.update(
                {
                    "status": "error",
                    "error": e.stderr or str(e),
                    "seconds": time.time() - t0,
                }
            )
        except FileNotFoundError as e:
            row.update(
                {
                    "status": "error",
                    "error": "mineru not found (pip install mineru)",
                    "seconds": time.time() - t0,
                }
            )
        except Exception as e:
            row.update(
                {"status": "error", "error": str(e), "seconds": time.time() - t0}
            )
        metrics.append(row)

    metrics_path = out_dir / "mineru_metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Saved: {metrics_path}")
    print(f"Saved markdown/outputs in: {out_dir}")


if __name__ == "__main__":
    main()
