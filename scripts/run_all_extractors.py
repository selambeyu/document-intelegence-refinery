#!/usr/bin/env python3
"""
Run pdfplumber, Docling, and MinerU on one or more PDFs.

Usage:
  python scripts/run_all_extractors.py path/to/file.pdf
  python scripts/run_all_extractors.py data/
  python scripts/run_all_extractors.py data/ -o .refinery/compare_out
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_pdfplumber(path: Path) -> dict:
    import pdfplumber
    out = {"path": str(path), "pages": [], "total_chars": 0, "total_area": 0.0, "image_area": 0.0}
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            area = float(page.width * page.height)
            chars = page.chars or []
            nchars = len(chars)
            img_area = 0.0
            if page.images:
                for im in page.images:
                    img_area += (im.get("width") or 0) * (im.get("height") or 0)
            out["total_chars"] += nchars
            out["total_area"] += area
            out["image_area"] += img_area
            out["pages"].append({
                "page": i + 1,
                "chars": nchars,
                "area_pt2": round(area, 2),
                "image_area": round(img_area, 2),
                "char_density": round(nchars / (area / 72 / 72), 2) if area else 0,
                "image_ratio": round(img_area / area, 4) if area else 0,
            })
    if out["total_area"]:
        out["overall_char_density"] = round(out["total_chars"] / (out["total_area"] / 72 / 72), 2)
        out["overall_image_ratio"] = round(out["image_area"] / out["total_area"], 4)
    return out


def run_docling(path: Path, out_dir: Path) -> Path:
    from docling.document_converter import DocumentConverter
    out_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(path)
    md_path = out_dir / f"{path.stem}_docling.md"
    md_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    return md_path


def run_mineru(path: Path, out_dir: Path) -> Path:
    out_dir = out_dir / "mineru"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["mineru", "-p", str(path), "-o", str(out_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    md_path = out_dir / f"{path.stem}.md"
    return md_path if md_path.exists() else out_dir


def main():
    parser = argparse.ArgumentParser(description="Run pdfplumber, Docling, MinerU on PDFs")
    parser.add_argument("input", type=Path, help="PDF file or directory")
    parser.add_argument("-o", "--output", type=Path, default=Path(".refinery/compare_out"), help="Output base dir")
    parser.add_argument("--no-mineru", action="store_true", help="Skip MinerU (CLI)")
    parser.add_argument("--no-docling", action="store_true", help="Skip Docling")
    args = parser.parse_args()

    base = args.input
    if not base.exists():
        print(f"Error: {base} does not exist", file=sys.stderr)
        sys.exit(1)
    pdfs = [base] if base.is_file() else sorted(base.glob("**/*.pdf"))
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        sys.exit(1)

    out_base = args.output
    out_base.mkdir(parents=True, exist_ok=True)

    for path in pdfs:
        print(f"\n--- {path.name} ---")
        plumber = run_pdfplumber(path)
        print("pdfplumber:", json.dumps({k: v for k, v in plumber.items() if k != "pages"}, indent=2))
        plumber_path = out_base / f"{path.stem}_pdfplumber.json"
        plumber_path.write_text(json.dumps(plumber, indent=2), encoding="utf-8")
        print(f"  -> {plumber_path}")

        if not args.no_docling:
            try:
                md_d = run_docling(path, out_base)
                print(f"Docling -> {md_d}")
            except Exception as e:
                print(f"Docling failed: {e}")

        if not args.no_mineru:
            try:
                run_mineru(path, out_base)
                print(f"MinerU -> {out_base / 'mineru'}")
            except FileNotFoundError:
                print("MinerU skipped: 'mineru' not found. Install with: uv sync --extra mineru  or  pip install mineru")
            except subprocess.CalledProcessError as e:
                print(f"MinerU failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
