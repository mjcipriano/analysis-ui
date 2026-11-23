"""Multi-graph page layouts: several graphs arranged in a grid with panel
labels (A, B, C …), exported as one figure.

    spec = {"panels": [{"graph": {...}, "data": {...}}, ...],
            "cols": 2, "width": 7.0, "panel_height": 2.8, "gap": 0.15,
            "labels": "A" | "a" | "1" | "none", "label_size": 14,
            "background": "#FFFFFF"}

Every panel is re-rendered at the size of its cell, so fonts and line widths
are identical across panels. SVG output is a vector composition; raster
formats (and PDF, which embeds the raster) are composed at the requested dpi.
"""
from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor

from .core.errors import AnalysisError
from .graphs.registry import render_graph

MAX_PANELS = 24


def _geometry(spec: dict):
    panels = spec.get("panels") or []
    if not panels:
        raise AnalysisError("Add at least one graph to the layout.")
    if len(panels) > MAX_PANELS:
        raise AnalysisError(f"A layout can hold at most {MAX_PANELS} graphs.")
    cols = max(1, min(int(spec.get("cols") or 2), len(panels), 8))
    width = min(max(float(spec.get("width") or 7.0), 2.0), 40.0)
    gap = min(max(float(spec.get("gap") if spec.get("gap") is not None else 0.15), 0.0), 2.0)
    ph = min(max(float(spec.get("panel_height") or 2.8), 1.0), 20.0)
    rows = -(-len(panels) // cols)
    cw = (width - gap * (cols - 1)) / cols
    if cw < 1.0:
        raise AnalysisError("The page is too narrow for this many columns.")
    height = rows * ph + gap * (rows - 1)
    return panels, cols, rows, width, height, cw, ph, gap


def _label(style: str, i: int) -> str:
    if style == "A":
        return chr(65 + i % 26)
    if style == "a":
        return chr(97 + i % 26)
    if style == "1":
        return str(i + 1)
    return ""


def _render_panels(panels, cw, ph, fmt, dpi):
    def one(p):
        g = dict(p.get("graph") or {})
        g["options"] = dict(g.get("options") or {}, width=round(cw, 4), height=round(ph, 4))
        return render_graph(g, p.get("data"), fmt=fmt, dpi=dpi)[0]
    # matplotlib renders are serialised by a lock; ggplot2 panels use the R pool
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(one, p) for p in panels]
        out = []
        for i, f in enumerate(futures):
            try:
                out.append(f.result())
            except AnalysisError as exc:
                raise AnalysisError(f"Panel {i + 1}: {exc}") from None
        return out


_ID = re.compile(r'\bid="([^"]+)"')


def _embed_svg(svg: str, prefix: str, x: float, y: float, w: float, h: float) -> str:
    """Nest one SVG document at (x, y) pt with size (w, h) pt, giving its ids a
    unique prefix so clip paths and glyph definitions of different panels do
    not collide."""
    svg = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", "", svg).strip()
    ids = set(_ID.findall(svg))
    if ids:
        svg = _ID.sub(lambda m: f'id="{prefix}{m.group(1)}"', svg)
        svg = re.sub(r'url\(#([^)]+)\)', lambda m: f"url(#{prefix}{m.group(1)})" if m.group(1) in ids else m.group(0), svg)
        svg = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{prefix}{m.group(1)}"' if m.group(1) in ids else m.group(0), svg)
    m = re.match(r"<svg\b([^>]*)>", svg)
    if not m:
        raise AnalysisError("A panel could not be composed (unexpected SVG).")
    attrs = m.group(1)
    vb = re.search(r'viewBox="([^"]+)"', attrs)
    keep = " ".join(re.findall(r'xmlns(?::\w+)?="[^"]+"', attrs))
    view = f' viewBox="{vb.group(1)}"' if vb else ""
    head = f'<svg {keep}{view} x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" preserveAspectRatio="xMidYMid meet">'
    return head + svg[m.end():]


def render_layout(spec: dict, fmt: str = "svg", dpi: int = 300) -> bytes:
    panels, cols, rows, width, height, cw, ph, gap = _geometry(spec)
    style = spec.get("labels") or "A"
    size = min(max(float(spec.get("label_size") or 14), 4.0), 60.0)
    bg = spec.get("background") or "#FFFFFF"
    transparent = bool(spec.get("transparent"))

    if fmt == "svg":
        parts = _render_panels(panels, cw, ph, "svg", dpi)
        W, H = width * 72, height * 72
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'width="{W:.2f}pt" height="{H:.2f}pt" viewBox="0 0 {W:.2f} {H:.2f}">']
        if not transparent:
            out.append(f'<rect width="100%" height="100%" fill="{bg}"/>')
        for i, svg in enumerate(parts):
            x = (i % cols) * (cw + gap) * 72
            y = (i // cols) * (ph + gap) * 72
            out.append(_embed_svg(svg.decode("utf-8"), f"p{i}-", x, y, cw * 72, ph * 72))
            lab = _label(style, i)
            if lab:
                out.append(f'<text x="{x + 3:.2f}" y="{y + size * 0.95:.2f}" font-family="Liberation Sans, Arial, Helvetica, sans-serif" '
                           f'font-weight="700" font-size="{size:.2f}" fill="{spec.get("label_color") or "#111111"}">{lab}</text>')
        out.append("</svg>")
        return "\n".join(out).encode("utf-8")

    if fmt not in ("png", "pdf", "tiff", "jpg"):
        raise AnalysisError(f"Unsupported image format: {fmt}")
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    dpi = int(min(max(dpi, 50), 1200))
    parts = _render_panels(panels, cw, ph, "png", dpi)
    W, H = round(width * dpi), round(height * dpi)
    if W * H > 250_000_000:
        raise AnalysisError("This layout is too large at the chosen resolution; lower the dpi.")
    rgba = fmt == "png" and transparent
    page = Image.new("RGBA" if rgba else "RGB", (W, H), (0, 0, 0, 0) if rgba else bg)
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(
            family=["Liberation Sans", "Arial", "DejaVu Sans"], weight="bold")), max(6, round(size * dpi / 72)))
    except Exception:
        font = ImageFont.load_default()
    pw, phh = round(cw * dpi), round(ph * dpi)
    for i, png in enumerate(parts):
        im = Image.open(io.BytesIO(png)).convert("RGBA")
        if im.size != (pw, phh):
            im = im.resize((pw, phh), Image.LANCZOS)
        x = round((i % cols) * (cw + gap) * dpi)
        y = round((i // cols) * (ph + gap) * dpi)
        page.paste(im, (x, y), im)
        lab = _label(style, i)
        if lab:
            draw.text((x + round(3 * dpi / 72), y + round(1 * dpi / 72)), lab, font=font,
                      fill=spec.get("label_color") or "#111111")
    buf = io.BytesIO()
    if fmt == "png":
        page.save(buf, "PNG", dpi=(dpi, dpi))
    elif fmt == "jpg":
        page.convert("RGB").save(buf, "JPEG", dpi=(dpi, dpi), quality=95)
    elif fmt == "tiff":
        page.convert("RGB").save(buf, "TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    else:
        page.convert("RGB").save(buf, "PDF", resolution=float(dpi))
    return buf.getvalue()
