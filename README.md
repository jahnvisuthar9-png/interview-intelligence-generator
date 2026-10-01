# Interview Intelligence Generator

A single-page web app that turns historical interview transcripts for one company and designation into the one-page Interview Intelligence Packet defined in `prompt.txt`.

## Put it online (free)

See **DEPLOY.md**: upload to GitHub, connect Render, paste your OpenAI key, share the link. No coding needed.

## Run it on your own computer

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

Inputs: company, designation and target round (required); optional **round info** (free text describing the round type/format, e.g. "Coding round, 2 DSA problems"); optional interview dates; optional transcripts (PDF, TXT, VTT, SRT, DOCX, multiple); optional JD as pasted text, a file, or both; optional resume.

### Two modes

- **Evidence-based** (transcripts supplied): everything in `prompt.txt` applies, including candidate-level counts like 7/10. Priority: transcripts > round info > JD/resume > round number. Questions come only from the transcripts; round info decides which rank first (coding questions for a coding round, design questions for system design). The round number is a secondary hint and is not shown on the PDF; the header shows the round type ("from round info" or "inferred from transcripts").
- **Role-based** (no transcripts): the model first decides the round type (from round info, or inferred from the round number and the company's known process), then builds a matching question mix (coding problems for coding rounds, design prompts for system design, behavioral for hiring-manager rounds), using the JD, resume and its knowledge of the company's usual questions for the role. `prompt.txt` is still sent, with an added note that its transcript rules can't apply. The page shows no counts or percentages; the header reads "ROLE-BASED | NO TRANSCRIPTS", section titles become "Expected round coverage" and "Likely questions", and the footer, Evidence_Matrix.csv and methodology all state that nothing was observed. QC verifies this labelling and that no counts appear.

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

## Optional settings

| Variable | Default | Use |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4.1` | Model name |
| `OPENAI_BASE_URL` | OpenAI | Point at any OpenAI-compatible API, e.g. Gemini's `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `MAX_PARALLEL_CALLS` | `4` | Set to `1` on free tiers with low per-minute limits |
| `OPENAI_FALLBACK_MODELS` | none | Comma-separated models to switch to when the main model's daily quota runs out or its name isn't found, e.g. `gemini-3.5-flash-lite,gemini-3.1-flash-lite` |
| `OPENAI_MAX_RETRIES` | `3` | Waits/retries for per-minute rate limits and brief outages. Daily-quota errors never wait; they switch model |
| `SINGLE_PASS_MAX_CHARS` | `800000` | Transcript batches up to this size (about 30-40 interviews) are analyzed in ONE model call; larger batches use per-interview calls |
| `ALWAYS_REVIEW_GROUPING` | off | Set to `1` to have the model review grouping (inside the same call) even when every file name has a candidate name |
| `OPENAI_REASONING_EFFORT` | unset | Optional `low`/`medium`/`high`; lower is faster but may reduce depth. Unset keeps the model default |

## API usage per packet

| Packet | Model calls |
|---|---|
| No transcripts (role-based) | 1 |
| Transcripts up to ~800,000 characters, clear or unclear file names | 1 (grouping, when needed, happens inside the same call) |
| Exactly the same inputs as an earlier run (same files, fields, prompt.txt, model) | 0 (earlier analysis reused for 24 hours) |
| Very large batches above the limit | 1 per interview + 1 (+1 if some names are unclear) |

Counts are always computed in code from the per-interview evidence the model returns. No fallback model is set by default, so every packet comes from `OPENAI_MODEL`; set `OPENAI_FALLBACK_MODELS` only if you accept a lighter model when the main one's daily quota runs out.

## Notes

- Scanned image-only PDFs have no extractable text; they're reported as unreadable rather than guessed at.
- Generated files are kept in `output/` for 24 hours, then deleted on the next run.
- Fonts: Inter (SIL Open Font License, see `fonts/Inter-LICENSE.txt`) with DejaVu Sans as glyph fallback.
