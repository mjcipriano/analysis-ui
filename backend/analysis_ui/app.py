"""HTTP API (FastAPI) and static hosting of the web app."""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, importer, transforms
from .core import registry
from .core.curvefit import MODELS
from .core.errors import AnalysisError
from .core.rbridge import pool, r_info
from .graphs import GRAPHS, registry as graph_registry, style

log = logging.getLogger("analysis_ui")
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
SAMPLES = HERE / "sample_data"

registry.discover()

app = FastAPI(title="Analysis UI", version=__version__)
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    pool().warm()


class RunRequest(BaseModel):
    analysis: str
    options: dict[str, Any] = {}
    data: dict[str, Any] | None = None


class GraphRequest(BaseModel):
    graph: dict[str, Any]
    data: dict[str, Any] | None = None
    format: str = "svg"
    dpi: int = 300
    editable_text: bool = False


class TextImport(BaseModel):
    text: str
    name: str = "Pasted data"


class TransformRequest(BaseModel):
    data: dict[str, Any]
    op: str
    columns: list[str] = []
    formula: str = ""
    name: str = ""
    id_column: str | None = None
    value_column: str | None = None


def _error(exc: Exception) -> dict:
    if isinstance(exc, AnalysisError):
        return {"status": "error", "message": str(exc)}
    log.exception("Unexpected error")
    return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/catalog")
def catalog():
    r = r_info()
    return {
        "version": __version__,
        "engines": {"python": True, "r": bool(r["available"]), "r_version": r.get("version")},
        "analysis_categories": registry.CATEGORIES,
        "analyses": [a.public() for a in registry.ANALYSES.values() if not a.id.startswith("_")],
        "graph_categories": graph_registry.CATEGORIES,
        "graphs": [g.public() for g in GRAPHS.values()],
        "palettes": style.palettes()["palettes"],
        "colormaps": style.palettes()["colormaps"],
        "themes": style.themes(),
        "transforms": transforms.TRANSFORMS,
        "curve_models": [{"id": m.id, "name": m.name, "equation": m.equation, "group": m.group}
                         for m in MODELS.values()],
        "import_extensions": importer.EXTENSIONS,
    }


@app.post("/api/analyses/run")
def run(req: RunRequest):
    return registry.run_analysis(req.analysis, req.data, req.options)


@app.post("/api/graphs/render")
def render(req: GraphRequest):
    """Preview: always returns JSON with an SVG string."""
    try:
        img, meta = graph_registry.render_graph(req.graph, req.data, fmt="svg")
        return {"status": "ok", "svg": img.decode("utf-8"), "meta": meta}
    except Exception as exc:
        return _error(exc)


@app.post("/api/graphs/export")
def export(req: GraphRequest):
    """Export: returns the image file itself."""
    try:
        img, _ = graph_registry.render_graph(req.graph, req.data, fmt=req.format, dpi=req.dpi,
                                             editable_text=req.editable_text)
    except Exception as exc:
        return JSONResponse(_error(exc), status_code=400)
    return Response(content=img, media_type=graph_registry.FORMATS[req.format])


@app.post("/api/import/file")
async def import_file(file: UploadFile = File(...)):
    try:
        return {"status": "ok", "sheets": importer.read_file(file.filename or "data.csv", await file.read())}
    except Exception as exc:
        return _error(exc)


@app.post("/api/import/text")
def import_text(req: TextImport):
    try:
        return {"status": "ok", "sheets": importer.read_text(req.text, req.name)}
    except Exception as exc:
        return _error(exc)


@app.post("/api/transform")
def transform(req: TransformRequest):
    try:
        if req.op in ("stack", "unstack"):
            return {"status": "ok", **transforms.reshape(req.data, req.op, req.columns, req.id_column, req.value_column)}
        return {"status": "ok", **transforms.transform(req.data, req.op, req.columns, req.formula, req.name)}
    except Exception as exc:
        return _error(exc)


@app.get("/api/samples")
def samples():
    index = SAMPLES / "index.json"
    return json.loads(index.read_text(encoding="utf-8")) if index.exists() else []


@app.get("/api/samples/{sample_id}")
def sample(sample_id: str):
    entry = next((s for s in samples() if s["id"] == sample_id), None)
    if entry is None:
        return JSONResponse({"status": "error", "message": "Unknown sample"}, status_code=404)
    try:
        sheets = importer.read_file(entry["file"], (SAMPLES / entry["file"]).read_bytes())
        sheets[0]["name"] = entry["name"]
        return {"status": "ok", "sheets": sheets, "sample": entry}
    except Exception as exc:
        return _error(exc)


# ---- the web app -----------------------------------------------------------
if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = (STATIC / path).resolve()
        if path and target.is_file() and STATIC in target.parents:
            return FileResponse(target)
        return FileResponse(STATIC / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def no_frontend():
        return JSONResponse({"message": "The API is running, but the web app has not been built. "
                                        "Run `npm install && npm run build` in frontend/."})
