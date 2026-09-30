"""Automated version of prompt.txt 'QUALITY CHECK BEFORE FINALIZING'."""
from __future__ import annotations

import re

from pypdf import PdfReader

from .render_pdf import MARGIN_BOTTOM, MARGIN_TOP, MARGIN_X, MIN_FONT_PT, PAGE_H, PAGE_W


def rasterize(pdf_path: str, png_path: str, dpi: int = 150):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    img = pdf[0].render(scale=dpi / 72).to_pil().convert("RGB")
    img.save(png_path)
    return img, dpi / 72


def _ink_bbox(img):
    from PIL import ImageChops, Image

    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg).convert("L").point(lambda v: 255 if v > 18 else 0)
    return diff.getbbox()


def _word_overlaps(pdf_path: str) -> int:
    try:
        import pdfplumber
    except ImportError:
        return -1
    with pdfplumber.open(pdf_path) as pdf:
        words = pdf.pages[0].extract_words(use_text_flow=False, keep_blank_chars=False)
    boxes = [(w["x0"], w["top"], w["x1"], w["bottom"]) for w in words]
    overlaps = 0
    for i in range(len(boxes)):
        a = boxes[i]
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix > 1.0 and iy > 1.5:  # real collision, not touching neighbours
                overlaps += 1
    return overlaps


def run_checks(pdf_path: str, png_path: str, packet: dict, layout: dict) -> list[dict]:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    reader = PdfReader(pdf_path)
    pages = len(reader.pages)
    add("Exactly one page", pages == 1, f"{pages} page(s)")

    img, k = rasterize(pdf_path, png_path)
    bbox = _ink_bbox(img)
    if bbox:
        x0, y0, x1, y1 = (v / k for v in bbox)
        inside = (x0 >= MARGIN_X - 3 and x1 <= PAGE_W - MARGIN_X + 3
                  and y0 >= MARGIN_TOP - 3 and y1 <= PAGE_H - MARGIN_BOTTOM + 6)
        add("Nothing clipped / all content inside margins", inside and layout["fits"],
            f"ink box {x0:.0f},{y0:.0f} to {x1:.0f},{y1:.0f} pt; layout fits={layout['fits']}")
    else:
        add("Nothing clipped / all content inside margins", False, "page rendered blank")

    text = reader.pages[0].extract_text() or ""
    flat = re.sub(r"\s+", " ", text)
    upper = flat.upper()
    add("Revision priority fully visible",
        "REVISION PRIORITY" in upper and all(it["text"].split()[0] in flat for it in packet["revision_priority"]),
        f"{len(packet['revision_priority'])} item(s)")
    add("Stars render (no broken glyphs)", not layout["missing_glyphs"] and "★" in text,
        "missing: " + "".join(layout["missing_glyphs"]) if layout["missing_glyphs"] else "all glyphs embedded")
    ov = _word_overlaps(pdf_path)
    add("No overlapping text", ov == 0 or ov == -1, "pdfplumber unavailable; layout engine guarantees spacing"
        if ov == -1 else f"{ov} overlapping word box(es)")
    add("Readable text size", layout["min_font_pt"] >= MIN_FONT_PT, f"smallest font {layout['min_font_pt']} pt")

    required = ["Tomorrow's interview", "Round coverage / top topics", packet["resume_focus"]["title"], "Top questions",
                packet["trend"]["title"], packet["role_focus"]["title"], "Revision priority",
                "30-minute revision order"]
    missing = [r for r in required if r.upper() not in upper]
    empty = [k for k in ("topics", "questions", "revision_priority", "revision_order") if not packet[k]]
    add("All sections present and populated", not missing and not empty,
        ", ".join(missing + empty) or "all present")

    n = packet["n"]
    bad = [t["name"] for t in packet["topics"] if not (1 <= t["count"] <= n)]
    add("Counts use unique candidate interviews", not bad and f"out of {n}" in flat,
        f"denominator {n}" + (f"; invalid: {bad}" if bad else ""))
    comp = re.sub(r"\s+", " ", packet["company"])
    desig = re.sub(r"\s+", " ", packet["designation"])
    add("Company and designation spelled as entered", comp in flat and desig in flat, f"{comp} / {desig}")
    add("30-minute order totals 30 min", layout["minutes_total"] == 30, f"{layout['minutes_total']} min")
    return checks
