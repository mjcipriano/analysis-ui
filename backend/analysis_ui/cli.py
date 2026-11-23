"""``analysis-ui`` command: start the server and open the browser."""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main():
    ap = argparse.ArgumentParser(prog="analysis-ui", description="Start Analysis UI.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    args = ap.parse_args()
    import uvicorn
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Analysis UI is running at {url}  (Ctrl+C to stop)")
    uvicorn.run("analysis_ui.app:app", host=args.host, port=args.port, reload=args.reload,
                log_level="warning")


if __name__ == "__main__":
    main()
