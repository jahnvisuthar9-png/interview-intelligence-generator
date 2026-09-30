"""End-to-end analysis: grouping -> per-interview extraction -> aggregate -> validated packet."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .grouping import InterviewGroup, TranscriptFile, build_files, finalize_groups, propose_groups
from .llm import (ANALYSIS_SCHEMA, APP_ADDENDUM, EXTRACTION_SCHEMA, GROUPING_SCHEMA,
                  load_source_prompt)

MAX_CHARS_PER_INTERVIEW = 600_000   # ~150k tokens; longer interviews are truncated with a note
SNIPPET_HEAD, SNIPPET_TAIL = 1500, 400


@dataclass
class Inputs:
    company: str
    designation: str
    target_round: str
    transcripts: list[tuple[str, str]]          # (filename, text)
    jd: tuple[str, str] | None = None           # (filename, text)
    resume: tuple[str, str] | None = None
    interview_dates: str = ""
    rejected_files: list[tuple[str, str]] = field(default_factory=list)  # (filename, reason)


def _system(stage_task: str) -> str:
    return load_source_prompt() + "\n" + APP_ADDENDUM + "\nSTAGE TASK:\n" + stage_task.strip()


# ------------------------------------------------------------------ stage prompts
GROUPING_TASK = """
Perform steps 1-6 of the CRITICAL TRANSCRIPT-HANDLING RULE only.
You receive every transcript file (id, file name, a deterministic name/part parse, whether it is
an exact duplicate, and the opening and closing text of the file). Exact duplicates are already
removed and must not be placed in any group.
Return one group per UNIQUE CANDIDATE INTERVIEW. Group primarily by Company + Candidate Name.
Treat suffixes such as -1, -2, -3, (2), part 2 as parts of the same interview unless the supplied
text clearly proves they are separate interview rounds (then use round_label, e.g. "Round 3";
otherwise round_label = ""). Only merge files with different parsed names when the text clearly
shows they are the same candidate's interview. candidate_label = normalized candidate first name
(or "Unknown N" if none can be found). Give a short reason for each group.
"""

EXTRACTION_TASK = """
Perform the per-interview part of the ANALYSIS WORKFLOW for ONE merged candidate interview
(all of its transcript parts are already merged below).
Ignore casual setup conversation (audio checks, location, greetings, scheduling, camera,
WhatsApp, connection issues, other non-interview talk).
List every substantive interviewer question, follow-up, probe, scenario, technical discussion,
behavioral probe and experience deep dive. Merge follow-ups into the question they belong to
(count them in follow_up_count). Normalize wording into a concise interview-ready pattern
("Explain your current AI/ML project end to end.") and give a short topic label.
experience_probes = which parts of the candidate's own experience were challenged
(e.g. exact contribution, production deployment, model selection, business impact).
round_evidence / date_evidence: only explicit mentions in the text, else "".
Do not include anything that was not actually asked.
"""

AGGREGATE_TASK = """
Perform sections A-G (CALCULATE THESE SECTIONS) across ALL merged candidate interviews, using the
per-interview extractions supplied. Interview ids (C1, C2, ...) are the UNIQUE CANDIDATE
INTERVIEWS; the app computes every count as (interviews listed) / (total interviews), so:
- topics (A): 6-8 recurring themes, normalized semantically. For each topic list EVERY interview
  id in which it genuinely appeared, with a short evidence phrase from that interview's
  extraction. One entry per interview at most.
- top_questions (B): 5-8 normalized question patterns that genuinely appeared; interview_ids =
  all interviews where the pattern appeared. Stars per prompt.txt (5 = highest). Max ~110 chars.
- resume_focus (C): 4-5 areas. If a resume is supplied, pick the parts of THIS resume most likely
  to be probed given interviewer behavior (basis cites both). If NOT supplied, only areas
  interviewers historically probed.
- role_focus (D): 4-6 areas. If a JD is supplied, JD responsibilities/skills ranked by how much
  the interviews emphasized them. If NOT supplied, observed role emphasis only.
- trend (E): chronological_analysis_possible = true only if reliable dates or chronological round
  ordering exist in the supplied dates or transcripts. 4-6 items.
- revision_priority (F): 4-6 items with stars; revision_order: 4-6 steps whose minutes sum to 30
  and reflect the interview evidence.
- tomorrow (G): review 3-4, practice_out_loud 3-4, remember 3-4 (short, evidence-based, no filler).
Keep every item short enough to scan: topics <= 55 chars, list items <= 70 chars.
Use plain text only (no star characters, no markdown, no percentages).
caveats: data-quality limitations worth noting in the methodology (may be empty).
"""


# ------------------------------------------------------------------ helpers
def _clamp_stars(v) -> int:
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return 3


_STRIP = re.compile(r"[★☆✓✔•■□▪●]+")


def _clean(s: str, limit: int = 160) -> str:
    s = _STRIP.sub("", str(s or "")).replace("**", "")
    s = re.sub(r"^\s*(\d{1,2}[.)]|[-–—])\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*-{1,2}>\s*", " → ", s)
    if len(s) > limit:
        s = s[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return s


def _normalize_minutes(steps: list[dict], total: int = 30) -> list[dict]:
    steps = [s for s in steps if s["item"]]
    if not steps:
        return steps
    raw = [max(1, int(s.get("minutes") or 1)) for s in steps]
    if sum(raw) == total:
        for s, m in zip(steps, raw):
            s["minutes"] = m
        return steps
    scale = total / sum(raw)
    exact = [r * scale for r in raw]
    mins = [max(1, int(x)) for x in exact]
    while sum(mins) < total:
        i = max(range(len(mins)), key=lambda k: exact[k] - mins[k])
        mins[i] += 1
    while sum(mins) > total:
        i = max((k for k in range(len(mins)) if mins[k] > 1), key=lambda k: mins[k] - exact[k])
        mins[i] -= 1
    for s, m in zip(steps, mins):
        s["minutes"] = m
    return steps


def _snippet(text: str) -> str:
    if len(text) <= SNIPPET_HEAD + SNIPPET_TAIL + 50:
        return text
    return text[:SNIPPET_HEAD] + "\n[...]\n" + text[-SNIPPET_TAIL:]


# ------------------------------------------------------------------ pipeline
def run(inputs: Inputs, llm, progress: Callable[[str], None] = lambda m: None) -> dict:
    notes: list[str] = []
    files = build_files(inputs.transcripts, inputs.company)
    if not files:
        raise ValueError("No readable transcript files were uploaded.")
    proposal = propose_groups(files)

    # ---- Stage 1: grouping review
    progress("Grouping transcript parts by candidate")
    grouping_payload = {
        "company": inputs.company,
        "files": [{
            "file_id": f.file_id, "file_name": f.filename,
            "parsed_candidate": f.proposed_candidate, "parsed_part_number": f.part_number,
            "exact_duplicate_of": f.duplicate_of,
            "text_preview": None if f.duplicate_of else _snippet(f.text),
        } for f in files],
        "deterministic_proposal": proposal,
    }
    grouping_source = "model-reviewed"
    try:
        g = llm.structured(_system(GROUPING_TASK), json.dumps(grouping_payload, ensure_ascii=False),
                           GROUPING_SCHEMA, "transcript_grouping")
        groups = finalize_groups(files, g.get("groups", []))
        notes += [_clean(n, 300) for n in g.get("notes", []) if n]
        group_reasons = {tuple(sorted(x.get("file_ids", []))): x.get("reason", "") for x in g.get("groups", [])}
    except Exception as e:  # fall back to the deterministic proposal, never to file count
        groups = finalize_groups(files, proposal)
        grouping_source = "deterministic (model review failed)"
        notes.append(f"Grouping review by the model failed ({e}); file-name grouping was used.")
        group_reasons = {}

    for grp in groups:
        if len(grp.merged_text) > MAX_CHARS_PER_INTERVIEW:
            grp.merged_text = grp.merged_text[:MAX_CHARS_PER_INTERVIEW]
            notes.append(f"{grp.interview_id} transcript was truncated to {MAX_CHARS_PER_INTERVIEW:,} characters.")

    # ---- Stage 2: per-interview extraction (parallel)
    progress(f"Analyzing {len(groups)} candidate interview(s)")

    def extract(grp: InterviewGroup) -> dict:
        user = (f"Company: {inputs.company}\nDesignation: {inputs.designation}\n"
                f"Target Round: {inputs.target_round}\nInterview id: {grp.interview_id}\n\n"
                f"MERGED TRANSCRIPT:\n{grp.merged_text}")
        out = llm.structured(_system(EXTRACTION_TASK), user, EXTRACTION_SCHEMA, "interview_extraction")
        out["interview_id"] = grp.interview_id
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        extractions = list(pool.map(extract, groups))

    kept_groups, kept_ex = [], []
    for grp, ex in zip(groups, extractions):
        if not ex.get("contains_real_interview_content") and not ex.get("questions"):
            notes.append(f"{grp.interview_id} ({', '.join(grp.file_ids)}) contained no interview "
                         "questions and was excluded from the denominator.")
            continue
        kept_groups.append(grp)
        kept_ex.append(ex)
    if not kept_groups:
        raise ValueError("None of the uploaded transcripts contained interview questions.")
    n = len(kept_groups)
    valid_ids = [g.interview_id for g in kept_groups]

    # ---- Stage 3: aggregate
    progress("Calculating topics, questions and priorities")
    dates_supplied = bool(inputs.interview_dates.strip())
    agg_user = json.dumps({
        "company": inputs.company,
        "designation": inputs.designation,
        "target_round": inputs.target_round,
        "unique_candidate_interviews": n,
        "interview_ids": valid_ids,
        "interview_dates": inputs.interview_dates.strip() or "NOT PROVIDED",
        "job_description": inputs.jd[1] if inputs.jd else "NOT PROVIDED",
        "candidate_resume": inputs.resume[1] if inputs.resume else "NOT PROVIDED",
        "per_interview_extractions": kept_ex,
    }, ensure_ascii=False)
    a = llm.structured(_system(AGGREGATE_TASK), agg_user, ANALYSIS_SCHEMA, "interview_intelligence")

    packet = validate(a, inputs, n, valid_ids, kept_ex, dates_supplied)
    packet["meta"].update({
        "transcript_files": [(f.file_id, f.filename) for f in files],
        "duplicates": [(f.filename, next(x.filename for x in files if x.file_id == f.duplicate_of))
                       for f in files if f.duplicate_of],
        "groups": [{"id": g.interview_id, "candidate": g.candidate_label, "round": g.round_label,
                    "files": [next(f.filename for f in files if f.file_id == fid) for fid in g.file_ids],
                    "reason": group_reasons.get(tuple(sorted(g.file_ids)), "")}
                   for g in kept_groups],
        "grouping_source": grouping_source,
        "notes": notes + [_clean(c, 300) for c in a.get("caveats", []) if c],
        "rejected_files": inputs.rejected_files,
        "jd_file": inputs.jd[0] if inputs.jd else None,
        "resume_file": inputs.resume[0] if inputs.resume else None,
        "model": getattr(llm, "model", "unknown"),
    })
    return packet


def validate(a: dict, inputs: Inputs, n: int, valid_ids: list[str], extractions: list[dict],
             dates_supplied: bool) -> dict:
    """Turn raw model output into a packet whose numbers are all derived in code."""
    valid = set(valid_ids)

    # A. topics: counts = unique valid interview ids with evidence
    topics = []
    for t in a.get("topics", []):
        occ: dict[str, str] = {}
        for o in t.get("occurrences", []):
            iid = str(o.get("interview_id", "")).strip()
            if iid in valid and iid not in occ:
                occ[iid] = _clean(o.get("evidence", ""), 200)
        name = _clean(t.get("name", ""), 60)
        if name and occ:
            topics.append({"name": name, "ids": [i for i in valid_ids if i in occ],
                           "evidence": occ, "count": len(occ)})
    topics.sort(key=lambda t: -t["count"])
    topics = topics[:8]

    # B. questions: only patterns with at least one real interview
    questions = []
    for q in a.get("top_questions", []):
        ids = [i for i in valid_ids if i in {str(x).strip() for x in q.get("interview_ids", [])}]
        text = _clean(q.get("question", ""), 130)
        if text and ids:
            questions.append({"text": text, "stars": _clamp_stars(q.get("stars")), "count": len(ids),
                              "ids": ids, "why": _clean(q.get("why_prioritized", ""), 200)})
    questions.sort(key=lambda q: -q["stars"])  # stable: keeps the model's priority order within a star level
    questions = questions[:8]

    def starred(items, key, limit, maxlen=80):
        out = []
        for it in items:
            txt = _clean(it.get(key, ""), maxlen)
            if txt:
                out.append({"text": txt, "stars": _clamp_stars(it.get("stars")),
                            "basis": _clean(it.get("basis", ""), 200)})
        out.sort(key=lambda x: -x["stars"])
        return out[:limit]

    resume_focus = starred(a.get("resume_focus", []), "area", 5)
    role_focus = starred(a.get("role_focus", []), "area", 6)
    revision_priority = starred(a.get("revision_priority", []), "item", 6)
    tomorrow = a.get("tomorrow", {})
    review = starred(tomorrow.get("review", []), "item", 4, 60)
    practice = starred(tomorrow.get("practice_out_loud", []), "item", 4, 60)
    remember = [_clean(r, 70) for r in tomorrow.get("remember", []) if _clean(r, 70)][:4]

    order = _normalize_minutes([{"item": _clean(s.get("item", ""), 80), "minutes": s.get("minutes", 1)}
                                for s in a.get("revision_order", [])][:6])

    # E. trend: chronological wording only when dates really exist
    trend = a.get("trend", {})
    transcript_dates = sum(1 for e in extractions if (e.get("date_evidence") or "").strip())
    chronological = bool(trend.get("chronological_analysis_possible")) and (dates_supplied or transcript_dates >= 2)
    trend_items = [{"topic": _clean(t.get("topic", ""), 60), "direction": t.get("direction", "stable")}
                   for t in trend.get("items", []) if _clean(t.get("topic", ""), 60)][:6]

    has_resume, has_jd = inputs.resume is not None, inputs.jd is not None
    return {
        "company": inputs.company.strip(),
        "designation": inputs.designation.strip(),
        "target_round": inputs.target_round.strip(),
        "n": n,
        "interview_ids": valid_ids,
        "topics": topics,
        "questions": questions,
        "resume_focus": {
            "title": "Resume focus" if has_resume else "Observed resume focus",
            "note": ("Based on this candidate's resume compared with what interviewers probed."
                     if has_resume else
                     "No candidate resume supplied; shows what interviewers historically probe in "
                     "candidate experience, not this candidate's resume."),
            "items": resume_focus},
        "role_focus": {
            "title": "JD focus" if has_jd else "Interview-derived role focus",
            "note": ("JD requirements ranked by how strongly interviews emphasized them."
                     if has_jd else
                     "Exact JD not supplied; rankings reflect observed interview emphasis."),
            "items": role_focus},
        "trend": {
            "title": "Recent trend" if chronological else "Preparation signal",
            "chronological": chronological,
            "note": (_clean(trend.get("reason", ""), 160) or "Direction based on interview dates.")
            if chronological else
            "Signals reflect recurrence and interviewer emphasis, not chronological change.",
            "items": trend_items},
        "revision_priority": revision_priority,
        "revision_order": order,
        "tomorrow": {"review": review, "practice": practice, "remember": remember},
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "dates_supplied": dates_supplied, "transcript_date_mentions": transcript_dates,
                 "trend_reason": _clean(trend.get("reason", ""), 300)},
    }
