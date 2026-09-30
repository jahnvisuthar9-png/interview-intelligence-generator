# Interview Intelligence Generator

A single-page web app that turns historical interview transcripts for one company and designation into the one-page Interview Intelligence Packet defined in `prompt.txt`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your key in .env
python app.py                                          # open http://127.0.0.1:5000
```

Or skip `.env` and export the key directly: `OPENAI_API_KEY=sk-... python app.py`.
The key is only read on the server; the browser never sees it. Set `OPENAI_MODEL` to change the model (default `gpt-4.1`; any model that supports Structured Outputs works).

## Flow

Upload files → Generate → OpenAI analysis (with `prompt.txt`) → one-page PDF → preview → download PDF or ZIP.

Inputs: company, designation, target round, optional interview dates, multiple transcripts (PDF, TXT, VTT, SRT, DOCX), optional JD, optional resume.

## How `prompt.txt` is applied

`prompt.txt` is loaded at runtime and sent **unmodified** as the system instructions of every OpenAI call. Each call adds only a short note saying which part of the prompt that call performs and what JSON shape to return (strict `json_schema` Structured Outputs).

| Stage | prompt.txt section | Where |
|---|---|---|
| Exact-duplicate removal (content hash), filename parsing, proposed grouping by Company + Candidate Name, suffixes -1/-2/(3)/part 2 treated as parts | Critical transcript-handling rule, steps 1–5 | `pipeline/grouping.py` |
| Model reviews the grouping using file text (can only split rounds when the text proves it); code validates every file is used exactly once | Steps 2–6 | `pipeline/analysis.py` |
| Per-interview extraction of real interviewer questions (setup chatter excluded), normalized patterns | Analysis workflow | `pipeline/analysis.py` (parallel) |
| Sections A–G across all interviews | Calculate these sections | `pipeline/analysis.py` |
| Counts, denominators, section titles, trend wording | Data accuracy | `validate()` in `pipeline/analysis.py` |
| One-page PDF | Design requirements, layout hierarchy | `pipeline/render_pdf.py` |
| Render-to-image and checks, auto-fix loop | Quality check before finalizing | `pipeline/qc.py`, `pipeline/package.py` |
| PDF, preview PNG, Evidence_Matrix.csv, Methodology_and_Notes.txt, ZIP | Output | `pipeline/package.py` |

Safeguards enforced in code, not left to the model:

- **Every count is computed** from the per-interview evidence the model lists, as `interviews / unique candidate interviews`. Unknown interview IDs are discarded; a question with no supporting interview is dropped.
- **Denominator** is the number of merged candidate interviews, never the number of files.
- **No resume** → section is titled "Observed resume focus" with the required disclaimer. **No JD** → "Interview-derived role focus" with "Exact JD not supplied; rankings reflect observed interview emphasis."
- **Trend** is only labelled chronological ("Recent trend") when the model says it is possible *and* dates were supplied or found in at least two transcripts. Otherwise it is "Preparation signal" with the required note.
- The 30-minute order is normalized to total exactly 30 minutes.

## One-page guarantee and QC

The renderer measures every block before drawing and searches for the largest type scale that fits one US Letter page (body text never below ~6.5 pt, notes floored at 6.2 pt). If content still doesn't fit, lists are trimmed to the lowest counts `prompt.txt` allows and the page is re-rendered. Then each PDF is rasterized and checked for: one page, no clipping (ink bounding box inside margins), Revision Priority visible, stars/glyphs embedded, no overlapping text (word bounding boxes), readable size, all sections present, correct denominator, company/designation spelling, 30-minute total. Results show in the app and in `Methodology_and_Notes.txt`.

## Test without an API key

```bash
python -m tests.test_offline
```

Uses a fake model response (clearly labelled test data) to exercise grouping, duplicate removal, counting, rendering, QC and packaging. Output lands in `tests/out/`.

```bash
python -m tests.compare_sample
```

Renders the reference sample's exact content through the app and compares it with `tests/reference_sample.pdf`: page count and size, every section heading, topic counts, and all QC checks. Writes `tests/compare/side_by_side.png` (reference left, app right).

## Notes

- Scanned image-only PDFs have no extractable text; they're reported as unreadable rather than guessed at.
- Generated files are kept in `output/` for 24 hours, then deleted on the next run.
- Fonts: Inter (SIL Open Font License, see `fonts/Inter-LICENSE.txt`) with DejaVu Sans as glyph fallback.
