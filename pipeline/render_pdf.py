"""One-page Interview Intelligence Packet renderer (ReportLab).

Layout follows the sample's hierarchy:
  HEADER -> TOMORROW'S INTERVIEW -> MAIN (left: coverage + resume | right: questions + signal)
  -> BOTTOM (role focus | revision priority) -> 30-MINUTE REVISION ORDER -> footer.
Every block is measured before drawing; a single scale factor is searched so the page always fits,
then any leftover height is shared out as breathing room between bands.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"

INK = HexColor("#16233B")        # strong dark accent
BLUE = HexColor("#2A52BE")       # primary highlight
MUTED = HexColor("#5E6A7E")
FAINT = HexColor("#8C96A8")
RULE = HexColor("#D9DEE7")
TINT = HexColor("#EEF2F9")
TRACK = HexColor("#E3E8F1")
AMBER = HexColor("#D99A06")
STAR_OFF = HexColor("#D3D8E1")
GREEN = HexColor("#1E8A57")
ON_DARK = HexColor("#FFFFFF")
ON_DARK_ACCENT = HexColor("#A9C0FF")
ON_DARK_FAINT = HexColor("#7F90B0")
ON_DARK_OFF = HexColor("#43557A")
RED = HexColor("#C0392B")

PAGE_W, PAGE_H = letter
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 36, 32, 26
CONTENT_W = PAGE_W - 2 * MARGIN_X
MIN_SCALE, MAX_SCALE = 0.80, 1.14
MIN_FONT_PT = 6.2

_FONTS_READY = False
_CMAPS: dict[str, set[int]] = {}


def register_fonts() -> None:
    global _FONTS_READY
    if _FONTS_READY:
        return
    for name, file in [("Inter", "Inter-Regular.ttf"), ("Inter-Medium", "Inter-Medium.ttf"),
                       ("Inter-SemiBold", "Inter-SemiBold.ttf"), ("Inter-Bold", "Inter-Bold.ttf"),
                       ("Inter-Italic", "Inter-Italic.ttf"), ("DejaVu", "DejaVuSans.ttf"),
                       ("DejaVu-Bold", "DejaVuSans-Bold.ttf")]:
        f = TTFont(name, str(FONT_DIR / file))
        pdfmetrics.registerFont(f)
        _CMAPS[name] = set(f.face.charToGlyph.keys())
    pdfmetrics.registerFontFamily("Inter", normal="Inter", bold="Inter-Bold",
                                  italic="Inter-Italic", boldItalic="Inter-Bold")
    _FONTS_READY = True


def glyph_report(text: str) -> tuple[str, list[str]]:
    """Return markup-safe text with DejaVu fallback for glyphs Inter lacks; list unrenderable chars."""
    register_fonts()
    out, missing, inter, dv = [], [], _CMAPS["Inter"], _CMAPS["DejaVu"]
    for ch in text:
        cp = ord(ch)
        if ch in "\n\t":
            out.append(" ")
        elif cp in inter or ch == " ":
            out.append(escape(ch))
        elif cp in dv:
            out.append(f'<font name="DejaVu">{escape(ch)}</font>')
        else:
            missing.append(ch)
    return "".join(out), missing


class Ctx:
    def __init__(self, scale: float):
        self.s = scale
        self.missing: set[str] = set()
        self.min_font = 99.0

    def fs(self, base: float) -> float:
        v = round(max(base * self.s, min(base, MIN_FONT_PT)), 2)
        self.min_font = min(self.min_font, v)
        return v

    def safe(self, text: str) -> str:
        markup, miss = glyph_report(text)
        self.missing.update(miss)
        return markup

    def style(self, size, font="Inter", color=INK, leading=1.28, align=0):
        size = self.fs(size)
        return ParagraphStyle("x", fontName=font, fontSize=size, leading=size * leading,
                              textColor=color, alignment=align, splitLongWords=True)


# ------------------------------------------------------------------ blocks
class Block:
    def height(self, w: float) -> float: ...
    def draw(self, c, x: float, top: float, w: float) -> None: ...


class Gap(Block):
    def __init__(self, h): self.h = h
    def height(self, w): return self.h
    def draw(self, c, x, top, w): pass


class Para(Block):
    def __init__(self, ctx, text, size, font="Inter", color=INK, markup=False, leading=1.28):
        self.p = Paragraph(text if markup else ctx.safe(text), ctx.style(size, font, color, leading))
    def height(self, w): return self.p.wrap(w, 10_000)[1]
    def draw(self, c, x, top, w):
        h = self.p.wrap(w, 10_000)[1]
        self.p.drawOn(c, x, top - h)


def tracked(c, x, base, text, font, size, color, space):
    """Letter-spaced text; state is saved/restored so tracking never leaks into later text."""
    c.saveState()
    t = c.beginText(x, base)
    t.setFont(font, size); t.setCharSpace(space); t.setFillColor(color)
    t.textOut(text)
    t.setCharSpace(0)
    c.drawText(t)
    c.restoreState()


class SectionHead(Block):
    """UPPERCASE tracked title + optional right-aligned hint; navy rule underneath (optional)."""
    def __init__(self, ctx, title, hint="", rule=True, color=INK, size=9.4):
        self.ctx, self.title, self.hint, self.rule, self.color = ctx, title.upper(), hint, rule, color
        self.size, self.hsize = ctx.fs(size), ctx.fs(7.0)
    def height(self, w): return self.size + (7 if self.rule else 1) * self.ctx.s
    def draw(self, c, x, top, w):
        base = top - self.size
        tracked(c, x, base, self.title, "Inter-Bold", self.size, self.color, 0.35 * self.ctx.s)
        title_w = stringWidth(self.title, "Inter-Bold", self.size) + 0.35 * self.ctx.s * len(self.title)
        hint_w = stringWidth(self.hint, "Inter-Italic", self.hsize) if self.hint else 0
        if self.hint and title_w + hint_w + 10 <= w:  # never let the hint touch the title
            c.setFillColor(FAINT); c.setFont("Inter-Italic", self.hsize)
            c.drawRightString(x + w, base, self.hint)
        if self.rule:
            c.setStrokeColor(INK); c.setLineWidth(0.9)
            y = base - 4.4 * self.ctx.s
            c.line(x, y, x + w, y)


def draw_stars(c, x, base, n, size, off_color=STAR_OFF):
    on, off = "★" * n, "☆" * (5 - n)
    c.setFont("Inter", size)
    c.setFillColor(AMBER); c.drawString(x, base, on)
    c.setFillColor(off_color); c.drawString(x + stringWidth(on, "Inter", size), base, off)


class LeadRow(Block):
    """Fixed-width lead (stars / arrow / number / check) + wrapping text."""
    def __init__(self, ctx, lead_kind, lead_val, text, size=8.1, color=INK, font="Inter",
                 lead_w=None, prefix_markup="", star_off=STAR_OFF, check_color=BLUE):
        self.ctx, self.kind, self.val = ctx, lead_kind, lead_val
        self.star_off, self.check_color = star_off, check_color
        self.size = ctx.fs(size)
        self.star_size = ctx.fs(6.9)
        if lead_w is None:
            lead_w = {"stars": stringWidth("★★★★★", "Inter", self.star_size) + 7 * ctx.s,
                      "arrow": 12 * ctx.s, "check": 11 * ctx.s}[lead_kind]
        self.lead_w = lead_w
        self.p = Paragraph(prefix_markup + ctx.safe(text), ctx.style(size, font, color))
    def height(self, w): return self.p.wrap(w - self.lead_w, 10_000)[1]
    def draw(self, c, x, top, w):
        h = self.p.wrap(w - self.lead_w, 10_000)[1]
        self.p.drawOn(c, x + self.lead_w, top - h)
        base = top - self.size
        if self.kind == "stars":
            draw_stars(c, x, base + 0.3, self.val, self.star_size, self.star_off)
        elif self.kind == "arrow":
            glyph, col = {"up": ("↑", GREEN), "down": ("↓", RED)}.get(self.val, ("→", FAINT))
            c.setFillColor(col); c.setFont("Inter-Bold", self.size)
            c.drawString(x, base, glyph)
        elif self.kind == "check":
            c.setFillColor(self.check_color); c.setFont("Inter-Bold", self.size)
            c.drawString(x, base, "✓")


class TopicRow(Block):
    """Topic text | frequency bar | n/N count, right-aligned."""
    def __init__(self, ctx, text, count, n):
        self.ctx, self.count, self.n = ctx, count, n
        self.size = ctx.fs(8.1)
        self.bar_w, self.num_w = 44 * ctx.s, 24 * ctx.s
        self.right = self.bar_w + self.num_w + 8 * ctx.s
        self.p = Paragraph(ctx.safe(text), ctx.style(8.1))
    def height(self, w): return self.p.wrap(w - self.right, 10_000)[1]
    def draw(self, c, x, top, w):
        h = self.p.wrap(w - self.right, 10_000)[1]
        self.p.drawOn(c, x, top - h)
        base = top - self.size
        bx = x + w - self.right + 8 * self.ctx.s
        by = base + 1.2 * self.ctx.s
        bh = 4.2 * self.ctx.s
        c.setFillColor(TRACK); c.roundRect(bx, by, self.bar_w, bh, bh / 2, stroke=0, fill=1)
        frac = self.count / self.n if self.n else 0
        if frac > 0:
            c.setFillColor(BLUE); c.roundRect(bx, by, max(bh, self.bar_w * frac), bh, bh / 2, stroke=0, fill=1)
        c.setFillColor(INK); c.setFont("Inter-SemiBold", self.size)
        c.drawRightString(x + w, base, f"{self.count}/{self.n}")


class Column:
    def __init__(self, blocks: list[Block], row_gap: float):
        self.blocks, self.row_gap = blocks, row_gap
    def height(self, w):
        hs = [b.height(w) for b in self.blocks]
        return sum(hs) + self.row_gap * max(0, len(hs) - 1)
    def draw(self, c, x, top, w):
        y = top
        for b in self.blocks:
            b.draw(c, x, y, w)
            y -= b.height(w) + self.row_gap


class Band:
    """Side-by-side columns; optional tinted background panel."""
    def __init__(self, columns: list[Column], gap: float, pad: float = 0, fill=None, weights=None,
                 title: Block | None = None, title_gap: float = 0):
        self.columns, self.gap, self.pad, self.fill = columns, gap, pad, fill
        self.weights = weights or [1] * len(columns)
        self.title, self.title_gap = title, title_gap
    def title_h(self):
        return self.title.height(CONTENT_W - 2 * self.pad) + self.title_gap if self.title else 0
    def widths(self):
        inner = CONTENT_W - 2 * self.pad - self.gap * (len(self.columns) - 1)
        tot = sum(self.weights)
        return [inner * wt / tot for wt in self.weights]
    def height(self):
        return (max(col.height(w) for col, w in zip(self.columns, self.widths()))
                + 2 * self.pad + self.title_h())
    def draw(self, c, top):
        h = self.height()
        if self.fill is not None:
            c.setFillColor(self.fill)
            c.roundRect(MARGIN_X, top - h, CONTENT_W, h, 7, stroke=0, fill=1)
        x = MARGIN_X + self.pad
        if self.title:
            self.title.draw(c, x, top - self.pad, CONTENT_W - 2 * self.pad)
        for col, w in zip(self.columns, self.widths()):
            col.draw(c, x, top - self.pad - self.title_h(), w)
            x += w + self.gap


# ------------------------------------------------------------------ page composition
def _star_rows(ctx, items, size=8.1, **kw):
    return [LeadRow(ctx, "stars", it["stars"], it["text"], size, **kw) for it in items]


def compose(packet: dict, ctx: Ctx):
    s, n = ctx.s, packet["n"]
    row = 3.0 * s

    # --- tomorrow's interview
    t = packet["tomorrow"]
    sub = lambda txt: SectionHead(ctx, txt, rule=False, color=ON_DARK_ACCENT, size=7.8)
    dark = dict(color=ON_DARK, star_off=ON_DARK_OFF)
    tomorrow_title = SectionHead(ctx, "Tomorrow's interview", rule=False, color=ON_DARK, size=9.4)
    cols = [
        Column([sub("Review")] + _star_rows(ctx, t["review"], 7.9, **dark), row),
        Column([sub("Practice out loud")] + _star_rows(ctx, t["practice"], 7.9, **dark), row),
        Column([sub("Remember")] + [LeadRow(ctx, "check", 0, r, 7.9, color=ON_DARK,
                                            check_color=ON_DARK_ACCENT) for r in t["remember"]], row),
    ]
    tomorrow = Band(cols, gap=16 * s, pad=12 * s, fill=INK, weights=[1.05, 1.05, 1],
                    title=tomorrow_title, title_gap=6 * s)

    # --- main left
    role = packet.get("mode") == "role"
    if role:
        left = [SectionHead(ctx, "Expected round coverage", "predicted, not observed"), Gap(2 * s)]
        left += _star_rows(ctx, [{"text": tp["name"], "stars": tp["stars"]} for tp in packet["topics"]])
    else:
        left = [SectionHead(ctx, "Round coverage / top topics", f"candidate-level, out of {n}"), Gap(2 * s)]
        left += [TopicRow(ctx, tp["name"], tp["count"], n) for tp in packet["topics"]]
    rf = packet["resume_focus"]
    left += [Gap(9 * s), SectionHead(ctx, rf["title"]), Gap(2 * s)] + _star_rows(ctx, rf["items"])
    left += [Gap(1 * s), Para(ctx, rf["note"], 6.7, "Inter-Italic", FAINT)]

    # --- main right
    right = [SectionHead(ctx, "Likely questions" if role else "Top questions",
                         "predicted, rehearse these" if role else "rehearse these"), Gap(2 * s)]
    for i, q in enumerate(packet["questions"], 1):
        num = f'<font name="Inter-SemiBold" color="#8C96A8">{i:02d}</font>&nbsp;&nbsp;'
        right.append(LeadRow(ctx, "stars", q["stars"], q["text"], 8.1, prefix_markup=num))
    tr = packet["trend"]
    hint = ("expected emphasis" if role else
            "by interview date" if tr["chronological"] else "recurrence, not chronology")
    right += [Gap(9 * s), SectionHead(ctx, tr["title"], hint), Gap(2 * s)]
    right += [LeadRow(ctx, "arrow", it["direction"], it["topic"]) for it in tr["items"]]
    right += [Gap(1 * s), Para(ctx, tr["note"], 6.7, "Inter-Italic", FAINT)]
    main = Band([Column(left, row), Column(right, row)], gap=24 * s)

    # --- bottom: role focus | revision priority
    ro = packet["role_focus"]
    b_left = [SectionHead(ctx, ro["title"]), Gap(2 * s)] + _star_rows(ctx, ro["items"])
    b_left += [Gap(1 * s), Para(ctx, ro["note"], 6.7, "Inter-Italic", FAINT)]
    b_right = [SectionHead(ctx, "Revision priority"), Gap(2 * s)] + _star_rows(ctx, packet["revision_priority"])
    bottom = Band([Column(b_left, row), Column(b_right, row)], gap=24 * s)

    # --- 30-minute order
    lead_w = 58 * s
    order_rows = [SectionHead(ctx, "30-minute revision order", rule=False), Gap(2 * s)]
    for i, st in enumerate(packet["revision_order"], 1):
        pre = (f'<font name="Inter-Bold" color="#2A52BE">{i:02d}</font>&nbsp;&nbsp;'
               f'<font name="Inter-Medium" color="#5E6A7E">{st["minutes"]} min</font>')
        order_rows.append(_OrderRow(ctx, pre, st["item"], lead_w))
    total = sum(st["minutes"] for st in packet["revision_order"])
    order = Band([Column(order_rows, row)], gap=0, pad=10 * s, fill=TINT)

    if role:
        footer_text = (f"Generated {packet['meta']['generated']}. No historical interview transcripts were "
                       f"supplied: this packet is predicted from the {packet['source_label']}, not observed "
                       f"interview data. Stars show expected preparation priority; no counts are shown because "
                       f"none were observed.")
    else:
        footer_text = None
    footer = Para(ctx, footer_text or
                  f"Generated {packet['meta']['generated']}. Denominator = {n} unique candidate "
                  f"interview{'s' if n != 1 else ''} (transcript parts merged per candidate, exact "
                  f"duplicates removed). Counts show interviews in which a topic appeared; stars show "
                  f"preparation priority. Built only from supplied transcripts"
                  f"{', JD' if packet['meta'].get('jd_file') else ''}"
                  f"{' and resume' if packet['meta'].get('resume_file') else ''}.",
                  6.4, "Inter", FAINT)
    return tomorrow, main, bottom, order, footer, total


class _OrderRow(Block):
    def __init__(self, ctx, lead_markup, text, lead_w):
        self.lead = Paragraph(lead_markup, ctx.style(8.1))
        self.p = Paragraph(ctx.safe(text), ctx.style(8.1))
        self.lead_w = lead_w
    def height(self, w): return self.p.wrap(w - self.lead_w, 10_000)[1]
    def draw(self, c, x, top, w):
        h = self.p.wrap(w - self.lead_w, 10_000)[1]
        lh = self.lead.wrap(self.lead_w, 1000)[1]
        self.lead.drawOn(c, x, top - lh)
        self.p.drawOn(c, x + self.lead_w, top - h)


def _subtitle(packet, ctx):
    sep = '&nbsp;&nbsp;<font color="#B4BCCB">|</font>&nbsp;&nbsp;'
    line2 = sep.join([ctx.safe(packet["company"]), ctx.safe(packet["designation"]),
                      ctx.safe("Target: " + packet["target_round"])])
    p = Paragraph(line2, ctx.style(10, "Inter-Medium", MUTED))
    pw = CONTENT_W
    return p, pw, p.wrap(pw, 1000)[1]


def header_height(packet, ctx):
    return ctx.fs(21) + 12 * ctx.s + _subtitle(packet, ctx)[2] + 1


def _draw_header(c, packet, ctx, top):
    s = ctx.s
    big = ctx.fs(21)
    base = top - big
    tracked(c, MARGIN_X, base, "INTERVIEW INTELLIGENCE", "Inter-Bold", big, INK, 0.2 * s)
    p, pw, ph = _subtitle(packet, ctx)
    p.drawOn(c, MARGIN_X, base - 6 * s - ph)
    # interview count, right aligned
    if packet.get("mode") == "role":
        lbl = "ROLE-BASED  |  NO TRANSCRIPTS"
    else:
        lbl = (f"{packet['n']} CANDIDATE INTERVIEW ANALYZED" if packet["n"] == 1
               else f"{packet['n']} CANDIDATE INTERVIEWS ANALYZED")
    lsize = ctx.fs(9)
    lw = stringWidth(lbl, "Inter-Bold", lsize) + 0.3 * s * (len(lbl) - 1)
    tracked(c, MARGIN_X + CONTENT_W - lw, top - lsize - 2 * s, lbl, "Inter-Bold", lsize, BLUE, 0.3 * s)
    y = base - 6 * s - ph - 6 * s
    c.setStrokeColor(INK); c.setLineWidth(1.6)
    c.line(MARGIN_X, y, MARGIN_X + CONTENT_W, y)
    return top - y


def layout_height(packet: dict, scale: float) -> tuple[float, Ctx]:
    ctx = Ctx(scale)
    tomorrow, main, bottom, order, footer, _ = compose(packet, ctx)
    gaps = 4 * 13 * scale
    total = (header_height(packet, ctx) + tomorrow.height() + main.height() + bottom.height() + order.height()
             + footer.height(CONTENT_W) + gaps + 8 * scale)
    return total, ctx


def render(packet: dict, out_path: str) -> dict:
    """Render the packet; returns layout facts used by QC."""
    register_fonts()
    avail = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
    lo, hi = MIN_SCALE, MAX_SCALE
    if layout_height(packet, lo)[0] > avail:
        scale = lo
    elif layout_height(packet, hi)[0] <= avail:
        scale = hi
    else:
        for _ in range(11):  # 0.34 / 2^11 < 0.0002 scale precision
            mid = (lo + hi) / 2
            if layout_height(packet, mid)[0] <= avail:
                lo = mid
            else:
                hi = mid
        scale = lo
    total, _ = layout_height(packet, scale)
    fits = total <= avail + 0.01
    spare = max(0.0, avail - total)

    ctx = Ctx(scale)
    tomorrow, main, bottom, order, footer, minutes_total = compose(packet, ctx)
    gap = 13 * scale + min(spare / 5, 14)

    c = canvas.Canvas(out_path, pagesize=letter)
    title = f"{packet['company']} {packet['designation']} Interview Intelligence Packet"
    c.setTitle(title); c.setAuthor("Interview Intelligence Generator"); c.setSubject(f"Target: {packet['target_round']}")
    top = PAGE_H - MARGIN_TOP
    top -= _draw_header(c, packet, ctx, top) + gap * 0.85
    tomorrow.draw(c, top); top -= tomorrow.height() + gap
    main.draw(c, top); top -= main.height() + gap
    bottom.draw(c, top); top -= bottom.height() + gap
    order.draw(c, top); top -= order.height() + 7 * scale
    footer.draw(c, MARGIN_X, top, CONTENT_W); top -= footer.height(CONTENT_W)
    c.showPage()
    c.save()
    return {"scale": round(scale, 3), "fits": fits and top >= MARGIN_BOTTOM - 0.5,
            "bottom_y": round(top, 1), "min_font_pt": round(ctx.min_font, 2),
            "missing_glyphs": sorted(ctx.missing), "minutes_total": minutes_total}
