"""A small pool of persistent R worker processes.

Each worker is ``Rscript r/worker.R``. Requests and replies travel through
temporary JSON files; only the file path and a sentinel line go over the
pipes, so chatter that R packages print can never corrupt a reply.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from .errors import AnalysisError

log = logging.getLogger("analysis_ui.r")

R_DIR = Path(__file__).resolve().parent.parent / "r"
SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"
DEFAULT_TIMEOUT = float(os.environ.get("AUI_R_TIMEOUT", "120"))
POOL_SIZE = int(os.environ.get("AUI_R_WORKERS", "2"))


class RUnavailable(AnalysisError):
    pass


def rscript_path() -> str | None:
    return os.environ.get("AUI_RSCRIPT") or shutil.which("Rscript")


class _Worker:
    def __init__(self):
        exe = rscript_path()
        if not exe:
            raise RUnavailable(
                "R is not installed (Rscript was not found), so analyses and "
                "graphs that use the R engine are unavailable.")
        self.tmp = Path(tempfile.mkdtemp(prefix="aui-r-"))
        env = dict(os.environ)
        env["AUI_R_DIR"] = str(R_DIR)
        env["AUI_SHARED_DIR"] = str(SHARED_DIR)
        # a UTF-8 locale so that µ, ρ, − ... survive in labels and tables
        if "UTF-8" not in (env.get("LC_ALL") or env.get("LC_CTYPE") or env.get("LANG") or "").upper().replace("UTF8", "UTF-8"):
            env["LC_ALL"] = "C.UTF-8"
        self.proc = subprocess.Popen(
            [exe, "--vanilla", str(R_DIR / "worker.R")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env, cwd=str(self.tmp))
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        self.stderr_tail: list[str] = []
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    def _pump_out(self):
        assert self.proc.stdout
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def _pump_err(self):
        assert self.proc.stderr
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip("\n"))
            del self.stderr_tail[:-40]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self):
        try:
            self.proc.kill()
        except Exception:  # pragma: no cover
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, request: dict, timeout: float) -> dict:
        rid = uuid.uuid4().hex
        req_path = self.tmp / f"{rid}.req.json"
        out_path = self.tmp / f"{rid}.out.json"
        request = dict(request, id=rid, out=str(out_path))
        req_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        try:
            assert self.proc.stdin
            self.proc.stdin.write(str(req_path) + "\n")
            self.proc.stdin.flush()
            sentinel = f"@@AUI-DONE {rid}"
            while True:
                try:
                    line = self.lines.get(timeout=timeout)
                except queue.Empty:
                    self.kill()
                    raise AnalysisError(
                        f"The R engine did not finish within {int(timeout)} seconds and was stopped.")
                if line is None:
                    tail = "\n".join(self.stderr_tail[-8:])
                    raise AnalysisError("The R engine stopped unexpectedly.\n" + tail)
                if line == sentinel:
                    break
            return json.loads(out_path.read_text(encoding="utf-8"))
        finally:
            for p in (req_path, out_path):
                try:
                    p.unlink()
                except OSError:
                    pass


class RPool:
    def __init__(self, size: int = POOL_SIZE):
        self.size = max(1, size)
        self._idle: "queue.LifoQueue[_Worker]" = queue.LifoQueue()
        self._count = 0
        self._lock = threading.Lock()

    def _acquire(self) -> _Worker:
        while True:
            try:
                w = self._idle.get_nowait()
            except queue.Empty:
                break
            if w.alive():
                return w
            with self._lock:
                self._count -= 1
        with self._lock:
            if self._count < self.size:
                self._count += 1
                create = True
            else:
                create = False
        if create:
            try:
                return _Worker()
            except Exception:
                with self._lock:
                    self._count -= 1
                raise
        w = self._idle.get()
        if not w.alive():
            with self._lock:
                self._count -= 1
            return self._acquire()
        return w

    def _release(self, w: _Worker):
        if w.alive():
            self._idle.put(w)
        else:
            with self._lock:
                self._count -= 1

    def call(self, request: dict, timeout: float | None = None) -> Any:
        w = self._acquire()
        try:
            reply = w.call(request, timeout or DEFAULT_TIMEOUT)
        finally:
            self._release(w)
        if reply.get("status") != "ok":
            msg = reply.get("message") or "Unknown R error"
            if isinstance(msg, list):
                msg = "\n".join(str(m) for m in msg)
            raise AnalysisError(str(msg).strip())
        return reply.get("result")

    def warm(self):
        """Start one worker in the background so the first request is fast."""
        def _go():
            try:
                self.call({"kind": "ping"}, timeout=60)
            except Exception as exc:  # pragma: no cover
                log.info("R warm-up failed: %s", exc)
        threading.Thread(target=_go, daemon=True).start()

    def shutdown(self):
        while True:
            try:
                self._idle.get_nowait().kill()
            except queue.Empty:
                break


_pool: RPool | None = None
_info: dict | None = None


def pool() -> RPool:
    global _pool
    if _pool is None:
        import atexit
        _pool = RPool()
        atexit.register(_pool.shutdown)
    return _pool


def r_info() -> dict:
    """``{"available": bool, "version": str, "packages": {name: bool}}``"""
    global _info
    if _info is None:
        if not rscript_path():
            _info = {"available": False, "version": None, "packages": {}}
        else:
            try:
                _info = dict(pool().call({"kind": "info"}, timeout=90), available=True)
            except Exception as exc:
                log.warning("R is installed but the worker failed: %s", exc)
                _info = {"available": False, "version": None, "packages": {}, "error": str(exc)}
    return _info


def r_has(*packages: str) -> bool:
    info = r_info()
    return bool(info["available"]) and all(info["packages"].get(p) for p in packages)
