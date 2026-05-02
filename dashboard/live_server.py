from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dashboard.export_aiops_stream import build_aiops_stream


DEFAULT_LIMIT = 500


def create_handler(
    *,
    trace_dir: str | Path = "traces",
    dashboard_dir: str | Path | None = None,
    limit: int = DEFAULT_LIMIT,
    algorithm: str = "auto",
) -> type[SimpleHTTPRequestHandler]:
    trace_path = Path(trace_dir)
    dashboard_path = Path(dashboard_dir) if dashboard_dir is not None else Path(__file__).resolve().parent
    default_limit = max(0, int(limit))

    class LiveDashboardHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(dashboard_path), **kwargs)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/aiops-stream.json":
                self._send_aiops_stream(parsed.query)
                return
            super().do_GET()

        def _send_aiops_stream(self, query: str) -> None:
            requested_limit = _limit_from_query(query, default_limit)
            stream = build_aiops_stream(
                trace_dir=trace_path,
                algorithm=algorithm,
                latest_only=True,
                limit=requested_limit,
            )
            body = json.dumps(stream, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return LiveDashboardHandler


def _limit_from_query(query: str, default: int) -> int:
    values = parse_qs(query).get("limit", [])
    if not values:
        return default
    try:
        return max(0, int(values[0]))
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the AIOps dashboard with live trace polling.")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--algorithm", default="auto")
    args = parser.parse_args()

    handler = create_handler(
        trace_dir=args.trace_dir,
        limit=args.limit,
        algorithm=args.algorithm,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Serving AIOps dashboard at {url}")
    print(f"Reading latest AIOps trace from {Path(args.trace_dir)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AIOps dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
