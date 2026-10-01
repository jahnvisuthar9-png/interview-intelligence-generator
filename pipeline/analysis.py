"""End-to-end analysis: grouping -> per-interview extraction -> aggregate -> validated packet."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .grouping import InterviewGroup, TranscriptFile, build_files, finalize_groups, propose_groups
import os

from .llm import (ANALYSIS_SCHEMA, APP_ADDENDUM, EXTRACTION_SCHEMA, GROUP_AND_ANALYZE_SCHEMA,
                  GROUPING_SCHEMA, ROLE_MODE_ADDENDUM, ROLE_SCHEMA, SINGLE_PASS_SCHEMA,
                  load_source_prompt)

MAX_CHARS_PER_INTERVIEW = 600_000   # ~150k tokens; longer interviews are truncated with a note
SNIPPET_HEAD, SNIPPET_TAIL = 1500, 400


@dataclass
class Inputs:
    company: str
    designation: str
    target_round: str
    transcripts: list[tuple[str, str]]          # (filename, text)
    jd: tuple[str, str] | None = None           # (source label, text): file name and/or "pasted text"
    resume: tuple[str, str] | None = None
    interview_dates: str = ""
    round_info: str = ""                        # optional free text: type/format of the round
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
round_evidence: which round this interview was (e.g. "Round 2", "initial screen", "final",
"hiring manager"), from explicit mentions in the text or the transcript file names, else "".
date_evidence: only explicit date mentions in the text, else "".
Do not include anything that was not actually asked.
"""

AGGREGATE_TASK = """
Perform sections A-G (CALCULATE THESE SECTIONS) across ALL merged candidate interviews, using the
per-interview extractions supplied. Use EVERY supplied input: company, designation, ROUND INFO, interview dates, JD
and resume (when supplied), alongside the transcripts.
PRIORITY ORDER: (1) the transcripts, (2) ROUND INFO, (3) JD and resume, (4) TARGET ROUND number.
- Questions and topics must be patterns that genuinely appeared in the transcripts.
- ROUND INFO (when supplied) decides which transcript questions matter most: for a coding round
  put coding/DSA problems and technical questions first; for system design put design questions
  first; for behavioral / hiring-manager rounds put behavioral and experience probes first; and so
  on. Use it the same way for topics, tomorrow and revision priorities.
- The TARGET ROUND number is only a secondary hint; do not let it override the transcripts or the
  round info. Each interview's round evidence is supplied; prefer interviews of the same round type.
- round_type: the type of round this packet prepares for. label = short name such as "Coding
  round", "System design round", "Technical screen", "Behavioral / hiring manager round";
  source = "round info" if ROUND INFO was supplied, otherwise "transcripts" (inferred from what
  the transcripts show). Interview ids (C1, C2, ...) are the UNIQUE CANDIDATE
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


SINGLE_PASS_TASK = """
Perform the ANALYSIS WORKFLOW and sections A-G in ONE pass over ALL merged candidate interviews
supplied (each interview's transcript parts are already merged; ids C1, C2, ... are the UNIQUE
CANDIDATE INTERVIEWS).
Read every interview in full. Ignore casual setup conversation (audio checks, location,
greetings, scheduling, camera, WhatsApp, connection issues, other non-interview talk). Focus on
interviewer questions, follow-ups, probes, scenarios, technical discussions, behavioral probes
and experience deep dives, and normalize semantically equivalent questions.
Counts are computed by the app from the interview ids you list, so for EVERY topic and question
check EVERY interview: a missed interview is a wrong count, and an id listed without real
evidence is a false count.
""" + AGGREGATE_TASK.replace("Perform sections A-G (CALCULATE THESE SECTIONS) across ALL merged candidate interviews, using the\nper-interview extractions supplied.", "") + """
interviews: one entry per interview id with contains_real_interview_content (false only if the
transcript has no interview questions at all), round_evidence (which round it was, e.g. "Round 2",
"initial screen", "final", "hiring manager", from the text or file names, else "") and
date_evidence (explicit date mentions only, else "").
"""


GROUP_AND_ANALYZE_TASK = """
STEP 1 - GROUPING (steps 1-6 of the CRITICAL TRANSCRIPT-HANDLING RULE). You receive every
transcript FILE (exact duplicates already removed) with a deterministic name/part parse and a
proposed grouping. Some file names have no candidate name, so check the text (speaker names,
introductions, content continuity). Return one group per UNIQUE CANDIDATE INTERVIEW in "groups",
with interview ids C1, C2, ... in order. Treat suffixes such as -1, -2, (2), part 2 as parts of
the same interview unless the text clearly proves separate rounds (then set round_label).
Only merge files with different names when the text clearly shows the same candidate's
interview. Every file must be in exactly one group.

STEP 2 - ANALYSIS. Treat each group's files as one merged interview and use YOUR interview ids
from step 1 everywhere below.
""" + SINGLE_PASS_TASK.replace("""(each interview's transcript parts are already merged; ids C1, C2, ... are the UNIQUE
CANDIDATE INTERVIEWS).""", """(the UNIQUE CANDIDATE INTERVIEWS are the groups you formed in step 1).""")


def _remap_ids(a: dict, idmap: dict) -> dict:
    """Translate the model's interview ids to the app's validated ids; unknown ids are dropped."""
    for t in a.get("topics", []):
        t["occurrences"] = [dict(o, interview_id=idmap[str(o.get("interview_id"))])
                            for o in t.get("occurrences", []) if str(o.get("interview_id")) in idmap]
    for q in a.get("top_questions", []):
        q["interview_ids"] = [idmap[str(i)] for i in q.get("interview_ids", []) if str(i) in idmap]
    a["interviews"] = [dict(m, interview_id=idmap[str(m.get("interview_id"))])
                       for m in a.get("interviews", []) if str(m.get("interview_id")) in idmap]
    return a


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
CACHE_VERSION = "4"  # bump when analysis logic changes so old cached results are not reused


def cache_key(inputs: Inputs, llm) -> str:
    """Fingerprint of everything that can change the answer: inputs, prompt.txt, model, logic version."""
    import hashlib

    blob = json.dumps({
        "v": CACHE_VERSION,
        "prompt": load_source_prompt(),
        "model": getattr(llm, "model", ""),
        "fallbacks": os.environ.get("OPENAI_FALLBACK_MODELS", ""),
        "company": inputs.company.strip().lower(),
        "designation": inputs.designation.strip().lower(),
        "round": inputs.target_round.strip().lower(),
        "round_info": inputs.round_info.strip(),
        "dates": inputs.interview_dates.strip(),
        "transcripts": sorted([n, t] for n, t in inputs.transcripts),
        "jd": inputs.jd[1] if inputs.jd else None,
        "resume": inputs.resume[1] if inputs.resume else None,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run(inputs: Inputs, llm, progress: Callable[[str], None] = lambda m: None,
        cache_dir=None) -> dict:
    """Analyze, reusing an earlier result when the exact same inputs were already analyzed
    (same files, fields, prompt.txt and model): a repeat costs zero API calls."""
    from pathlib import Path

    path = Path(cache_dir) / f"{cache_key(inputs, llm)}.json" if cache_dir else None
    if path is not None and path.exists():
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
            progress("Same inputs as an earlier run: reusing that analysis (no API call)")
            packet["meta"]["notes"] = list(packet["meta"].get("notes", [])) + [
                f"Reused the analysis from an identical earlier request ({packet['meta']['generated']}); "
                "no new API call was made."]
            packet["meta"]["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            return packet
        except Exception:
            pass  # unreadable cache entry: analyze normally
    packet = _run(inputs, llm, progress)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return packet


def _run(inputs: Inputs, llm, progress: Callable[[str], None] = lambda m: None) -> dict:
    if not inputs.transcripts:
        return run_role_based(inputs, llm, progress)
    notes: list[str] = []
    files = build_files(inputs.transcripts, inputs.company)
    if not files:
        raise ValueError("No readable transcript files were uploaded.")
    proposal = propose_groups(files)

    # ---- Grouping. Clear file names (Google-Sohan-1, Google-Sohan-2) are grouped by the prompt.txt
    # rule directly. If some names are unclear, the model groups them INSIDE the single analysis
    # call, so the packet still costs one call.
    group_reasons: dict = {}
    unnamed = [f for f in files if not f.duplicate_of and not f.proposed_candidate]
    needs_review = bool(unnamed) or os.environ.get("ALWAYS_REVIEW_GROUPING") == "1"
    usable_chars = sum(min(len(f.text), MAX_CHARS_PER_INTERVIEW) for f in files if not f.duplicate_of)
    single_pass = usable_chars <= int(os.environ.get("SINGLE_PASS_MAX_CHARS", "800000"))

    dates_supplied = bool(inputs.interview_dates.strip())
    names_of = lambda grp: [f.filename for f in files if f.file_id in grp.file_ids]
    common = {
        "company": inputs.company,
        "designation": inputs.designation,
        "target_round": inputs.target_round,
        "round_info": inputs.round_info.strip() or "NOT PROVIDED",
        "interview_dates": inputs.interview_dates.strip() or "NOT PROVIDED",
        "job_description": inputs.jd[1] if inputs.jd else "NOT PROVIDED",
        "candidate_resume": inputs.resume[1] if inputs.resume else "NOT PROVIDED",
    }

    if single_pass and needs_review:
        # ---- One call: group the files AND analyze them.
        progress("Grouping and analyzing transcripts")
        live = [f for f in files if not f.duplicate_of]
        payload = dict(common, **{
            "files": [{"file_id": f.file_id, "file_name": f.filename,
                       "parsed_candidate": f.proposed_candidate or "UNKNOWN",
                       "parsed_part_number": f.part_number,
                       "text": f.text[:MAX_CHARS_PER_INTERVIEW]} for f in live],
            "exact_duplicates_removed": [f.filename for f in files if f.duplicate_of],
            "deterministic_proposal": proposal,
        })
        a = llm.structured(_system(GROUP_AND_ANALYZE_TASK), json.dumps(payload, ensure_ascii=False),
                           GROUP_AND_ANALYZE_SCHEMA, "interview_intelligence_grouped")
        model_groups = a.get("groups", [])
        groups = finalize_groups(files, model_groups)   # validated: every file used exactly once
        idmap = {}
        for g in groups:
            owner = next((mg for mg in model_groups if g.file_ids[0] in mg.get("file_ids", [])), None)
            if owner is not None and str(owner.get("interview_id")) not in idmap:
                idmap[str(owner.get("interview_id"))] = g.interview_id
            else:
                notes.append(f"{g.interview_id} ({', '.join(names_of(g))}) was not grouped by the model; "
                             "it was kept as its own interview with no topic evidence.")
        a = _remap_ids(a, idmap)
        group_reasons = {tuple(sorted(x.get("file_ids", []))): x.get("reason", "") for x in model_groups}
        grouping_source = "model-reviewed inside the analysis call (some file names had no candidate name)"
    else:
        if needs_review:  # large batch only: separate grouping call
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
            grouping_source = "model-reviewed (some file names had no candidate name)"
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
        else:
            groups = finalize_groups(files, proposal)
            grouping_source = "file names (every file had a clear candidate name, so no model review was needed)"

        for grp in groups:
            if len(grp.merged_text) > MAX_CHARS_PER_INTERVIEW:
                grp.merged_text = grp.merged_text[:MAX_CHARS_PER_INTERVIEW]
                notes.append(f"{grp.interview_id} transcript was truncated to {MAX_CHARS_PER_INTERVIEW:,} characters.")

        if single_pass:
            # ---- One call: read every merged interview and produce sections A-G with per-interview evidence.
            progress(f"Analyzing {len(groups)} candidate interview(s)")
            payload = dict(common, **{
                "unique_candidate_interviews": len(groups),
                "interview_ids": [g.interview_id for g in groups],
                "interview_rounds": [{"interview_id": g.interview_id,
                                      "round_label_from_grouping": g.round_label or "unknown",
                                      "file_names": names_of(g)} for g in groups],
                "interviews": [{"interview_id": g.interview_id, "file_names": names_of(g),
                                "merged_transcript": g.merged_text} for g in groups],
            })
            a = llm.structured(_system(SINGLE_PASS_TASK), json.dumps(payload, ensure_ascii=False),
                               SINGLE_PASS_SCHEMA, "interview_intelligence_single_pass")

    total_chars = usable_chars
    if single_pass:
        meta = {str(m.get("interview_id")): m for m in a.get("interviews", [])}
        kept_groups, kept_ex = [], []
        for grp in groups:
            m = meta.get(grp.interview_id, {})
            if m and not m.get("contains_real_interview_content", True):
                notes.append(f"{grp.interview_id} ({', '.join(names_of(grp))}) contained no interview "
                             "questions and was excluded from the denominator.")
                continue
            kept_groups.append(grp)
            kept_ex.append({"interview_id": grp.interview_id,
                            "round_evidence": m.get("round_evidence", ""),
                            "date_evidence": m.get("date_evidence", "")})
        if not kept_groups:
            raise ValueError("None of the uploaded transcripts contained interview questions.")
        n = len(kept_groups)
        valid_ids = [g.interview_id for g in kept_groups]
        analysis_mode = "single pass (all interviews analyzed together in one model call)"
    else:
        # ---- Large batches: per-interview extraction, then aggregate.
        progress(f"Analyzing {len(groups)} candidate interview(s) one by one (large batch)")

        def extract(grp: InterviewGroup) -> dict:
            user = (f"Company: {inputs.company}\nDesignation: {inputs.designation}\n"
                    f"Target Round: {inputs.target_round}\nInterview id: {grp.interview_id}\n"
                    f"Transcript file names: {', '.join(names_of(grp))}\n\n"
                    f"MERGED TRANSCRIPT:\n{grp.merged_text}")
            out = llm.structured(_system(EXTRACTION_TASK), user, EXTRACTION_SCHEMA, "interview_extraction")
            out["interview_id"] = grp.interview_id
            return out

        with ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("MAX_PARALLEL_CALLS", "4")))) as pool:
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

        progress("Calculating topics, questions and priorities")
        agg_user = json.dumps(dict(common, **{
            "unique_candidate_interviews": n,
            "interview_ids": valid_ids,
            "interview_rounds": [{"interview_id": g.interview_id,
                                  "round_label_from_grouping": g.round_label or "unknown",
                                  "round_evidence_in_transcript": e.get("round_evidence") or "none",
                                  "file_names": names_of(g)}
                                 for g, e in zip(kept_groups, kept_ex)],
            "per_interview_extractions": kept_ex,
        }), ensure_ascii=False)
        a = llm.structured(_system(AGGREGATE_TASK), agg_user, ANALYSIS_SCHEMA, "interview_intelligence")
        analysis_mode = f"per-interview extraction + aggregate ({total_chars:,} characters exceeded the one-call limit)"

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
        "analysis_mode": analysis_mode,
        "notes": notes + list(getattr(llm, "events", [])) + [_clean(c, 300) for c in a.get("caveats", []) if c],
        "rejected_files": inputs.rejected_files,
        "jd_file": inputs.jd[0] if inputs.jd else None,
        "resume_file": inputs.resume[0] if inputs.resume else None,
        "model": getattr(llm, "used_label", None) or getattr(llm, "model", "unknown"),
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

    shared = _shared_sections(a)

    # E. trend: chronological wording only when dates really exist
    trend = a.get("trend", {})
    transcript_dates = sum(1 for e in extractions if (e.get("date_evidence") or "").strip())
    chronological = bool(trend.get("chronological_analysis_possible")) and (dates_supplied or transcript_dates >= 2)
    trend_items = [{"topic": _clean(t.get("topic", ""), 60), "direction": t.get("direction", "stable")}
                   for t in trend.get("items", []) if _clean(t.get("topic", ""), 60)][:6]

    has_resume, has_jd = inputs.resume is not None, inputs.jd is not None
    return {
        "round_info": inputs.round_info.strip(),
        "round_type": _round_type(a, inputs, default_source="transcripts"),
        "company": inputs.company.strip(),
        "designation": inputs.designation.strip(),
        "target_round": inputs.target_round.strip(),
        "n": n,
        "interview_ids": valid_ids,
        "topics": topics,
        "questions": questions,
        "mode": "transcripts",
        "resume_focus": {
            "title": "Resume focus" if has_resume else "Observed resume focus",
            "note": ("Based on this candidate's resume compared with what interviewers probed."
                     if has_resume else
                     "No candidate resume supplied; shows what interviewers historically probe in "
                     "candidate experience, not this candidate's resume."),
            "items": shared["resume_focus"]},
        "role_focus": {
            "title": "JD focus" if has_jd else "Interview-derived role focus",
            "note": ("JD requirements ranked by how strongly interviews emphasized them."
                     if has_jd else
                     "Exact JD not supplied; rankings reflect observed interview emphasis."),
            "items": shared["role_focus"]},
        "trend": {
            "title": "Recent trend" if chronological else "Preparation signal",
            "chronological": chronological,
            "note": (_clean(trend.get("reason", ""), 160) or "Direction based on interview dates.")
            if chronological else
            "Signals reflect recurrence and interviewer emphasis, not chronological change.",
            "items": trend_items},
        "revision_priority": shared["revision_priority"],
        "revision_order": shared["revision_order"],
        "tomorrow": shared["tomorrow"],
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "dates_supplied": dates_supplied, "transcript_date_mentions": transcript_dates,
                 "trend_reason": _clean(trend.get("reason", ""), 300)},
    }


def _round_type(a: dict, inputs: Inputs, default_source: str) -> dict:
    rt = a.get("round_type") or {}
    label = _clean(rt.get("label", ""), 40)
    source = (rt.get("source") or default_source).strip().lower()
    if inputs.round_info.strip():
        source = "round info"
        if not label:
            label = _clean(inputs.round_info, 40)
    return {"label": label, "source": source}


def _starred(items, key, limit, maxlen=80):
    out = []
    for it in items:
        txt = _clean(it.get(key, ""), maxlen)
        if txt:
            out.append({"text": txt, "stars": _clamp_stars(it.get("stars")),
                        "basis": _clean(it.get("basis", ""), 200)})
    out.sort(key=lambda x: -x["stars"])
    return out[:limit]


def _shared_sections(a: dict) -> dict:
    """Sections built the same way in both modes (F, G, resume and role focus items)."""
    t = a.get("tomorrow", {})
    return {
        "resume_focus": _starred(a.get("resume_focus", []), "area", 5),
        "role_focus": _starred(a.get("role_focus", []), "area", 6),
        "revision_priority": _starred(a.get("revision_priority", []), "item", 6),
        "revision_order": _normalize_minutes(
            [{"item": _clean(x.get("item", ""), 80), "minutes": x.get("minutes", 1)}
             for x in a.get("revision_order", [])][:6]),
        "tomorrow": {
            "review": _starred(t.get("review", []), "item", 4, 60),
            "practice": _starred(t.get("practice_out_loud", []), "item", 4, 60),
            "remember": [_clean(r, 70) for r in t.get("remember", []) if _clean(r, 70)][:4],
        },
    }


# ------------------------------------------------------------------ role-based mode
ROLE_TASK = """
Build every packet section for the target round from the company, designation, round number,
ROUND INFO, any JD / resume supplied, and your own knowledge of how this company interviews for this
role and round (see ROLE-BASED MODE above).
FIRST decide round_type: if ROUND INFO is supplied, use it (source "round info"); otherwise infer
the round type from the round number (initial screen / round 1 / 2 / 3 / final) and this
company's known interview process for this role (source "round number" or "company process").
The round type then decides the question mix:
- coding round: concrete coding/DSA problems (name the problem type, e.g. "Two-sum variant on a
  stream", "LRU cache") plus technical follow-ups on complexity and edge cases;
- technical screen: core technical concepts for the role plus one light coding/problem question;
- system design round: design prompts relevant to this company's products and scale, plus
  trade-off follow-ups;
- behavioral / hiring manager / culture round: behavioral and ownership questions in this
  company's known style (e.g. its leadership principles or values);
- domain / case / product rounds: the case or domain questions typical for this role.
Use the JD details (skills, tools, responsibilities) to make questions specific, and this
company's known interview history and usual questions for this role.
- topics: 6-8 topics this round is most likely to cover for this designation, with stars for
  expected emphasis and a short basis (JD line, resume item, or the role/round itself).
- top_questions: 5-8 likely question patterns, concise and interview-ready, with stars.
- resume_focus: 4-5 areas. With a resume: the parts of THIS resume most likely to be probed.
  Without one: the experience areas this role/round typically probes.
- role_focus: 4-6 areas. With a JD: JD responsibilities/skills ranked by likely interview weight.
  Without one: the core competencies of the designation.
- signal: 4-6 topics with direction up (prepare most), stable (core), down (lower priority).
- revision_priority 4-6 with stars; revision_order 4-6 steps totalling 30 minutes.
- tomorrow: review 3-4, practice_out_loud 3-4, remember 3-4 (specific, no filler).
Topics <= 55 chars, list items <= 70 chars, questions <= 110 chars. Plain text, no star
characters, no counts or percentages.
caveats: limitations worth noting (e.g. no JD, no resume, no transcripts).
"""


def run_role_based(inputs: Inputs, llm, progress: Callable[[str], None] = lambda m: None) -> dict:
    """No transcripts: one model call grounded in designation, round, JD and resume."""
    progress("Building a role-based packet (no transcripts supplied)")
    system = load_source_prompt() + "\n" + APP_ADDENDUM + ROLE_MODE_ADDENDUM + "\nSTAGE TASK:\n" + ROLE_TASK.strip()
    user = json.dumps({
        "company": inputs.company, "designation": inputs.designation,
        "target_round": inputs.target_round,
        "round_info": inputs.round_info.strip() or "NOT PROVIDED",
        "job_description": inputs.jd[1] if inputs.jd else "NOT PROVIDED",
        "candidate_resume": inputs.resume[1] if inputs.resume else "NOT PROVIDED",
        "historical_interview_transcripts": "NOT PROVIDED",
    }, ensure_ascii=False)
    a = llm.structured(system, user, ROLE_SCHEMA, "role_based_packet")
    packet = validate_role(a, inputs)
    packet["meta"].update({
        "transcript_files": [], "duplicates": [], "groups": [], "grouping_source": "not applicable",
        "notes": list(getattr(llm, "events", [])) + [_clean(c, 300) for c in a.get("caveats", []) if c],
        "rejected_files": inputs.rejected_files,
        "jd_file": inputs.jd[0] if inputs.jd else None,
        "resume_file": inputs.resume[0] if inputs.resume else None,
        "model": getattr(llm, "used_label", None) or getattr(llm, "model", "unknown"),
    })
    return packet


def validate_role(a: dict, inputs: Inputs) -> dict:
    shared = _shared_sections(a)
    has_resume, has_jd = inputs.resume is not None, inputs.jd is not None
    parts = (["designation", "interview round"] + (["round info"] if inputs.round_info.strip() else [])
             + (["JD"] if has_jd else []) + (["resume"] if has_resume else []))
    source = (", ".join(parts[:-1]) + " and " + parts[-1] +
              f", plus the model's general knowledge of {inputs.company.strip()}'s interview style")
    topics = [{"name": _clean(t.get("name", ""), 60), "stars": _clamp_stars(t.get("stars")),
               "basis": _clean(t.get("basis", ""), 200)}
              for t in a.get("topics", []) if _clean(t.get("name", ""), 60)]
    topics.sort(key=lambda t: -t["stars"])
    questions = [{"text": _clean(q.get("question", ""), 130), "stars": _clamp_stars(q.get("stars")),
                  "count": 0, "ids": [], "why": _clean(q.get("why_prioritized", ""), 200)}
                 for q in a.get("top_questions", []) if _clean(q.get("question", ""), 130)]
    questions.sort(key=lambda q: -q["stars"])
    return {
        "round_info": inputs.round_info.strip(),
        "round_type": _round_type(a, inputs, default_source="round number"),
        "mode": "role",
        "source_label": source,
        "company": inputs.company.strip(),
        "designation": inputs.designation.strip(),
        "target_round": inputs.target_round.strip(),
        "n": 0,
        "interview_ids": [],
        "topics": topics[:8],
        "questions": questions[:8],
        "resume_focus": {
            "title": "Resume focus" if has_resume else "Expected resume focus",
            "note": ("Parts of this candidate's resume most likely to be probed for this role and round."
                     if has_resume else
                     "No resume supplied; typical experience areas probed for this role and round."),
            "items": shared["resume_focus"]},
        "role_focus": {
            "title": "JD focus" if has_jd else "Role focus",
            "note": ("JD requirements ranked by expected interview weight; not observed interview data."
                     if has_jd else
                     "No JD supplied; core competencies expected for this designation."),
            "items": shared["role_focus"]},
        "trend": {
            "title": "Preparation signal", "chronological": False,
            "note": "Signals reflect expected emphasis for this role and round, not observed or chronological data.",
            "items": [{"topic": _clean(t.get("topic", ""), 60), "direction": t.get("direction", "stable")}
                      for t in a.get("signal", []) if _clean(t.get("topic", ""), 60)][:6]},
        "revision_priority": shared["revision_priority"],
        "revision_order": shared["revision_order"],
        "tomorrow": shared["tomorrow"],
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "dates_supplied": False,
                 "transcript_date_mentions": 0, "trend_reason": ""},
    }
