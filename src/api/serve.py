"""Run the Refinery web UI and API. Usage: refinery-serve [--port 8000]"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Document Intelligence Refinery web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Reload on code change")
    args = parser.parse_args()
    try:
        import uvicorn
        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except ImportError:
        print("Run: pip install uvicorn[standard] (or uv sync)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
