"""Start the Refinery web UI. Run: python scripts/serve.py or uv run python scripts/serve.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.serve import main

if __name__ == "__main__":
    main()
