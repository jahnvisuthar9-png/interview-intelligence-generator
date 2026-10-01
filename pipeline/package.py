"""Build the deliverables named in prompt.txt OUTPUT: PDF, preview PNG, Evidence_Matrix.csv,
Methodology_and_Notes.txt and the ZIP package. Includes the fix-before-delivery loop."""
from __future__ import annotations

import copy
import csv
import io
import re
import zipfile
from pathlib import Path

from .qc import run_checks
from .render_pdf import render

# Progressive trims applied only if the page cannot fit at the smallest readable scale.
# Each keeps every list within the ranges prompt.txt asks for.
TRIM_STEPS = [
    {"questions": 7, "topics": 8, "resume": 5, "role": 5, "trend": 5, "priority": 5},
    {"questions": 6, "topics": 7, "resume": 4, "role": 5, "trend": 5, "priority": 5},
    {"questions": 6, "topics": 6, "resume": 4, "role": 4, "trend": 4, "priority": 4},
    {"questions": 5, "topics": 6, "resume": 4, "role": 4, "trend": 4, "priority": 4},
]


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")
    return s or "Unknown"


def base_name(packet: dict) -> str:
    return f"{slug(packet['company'])}_{slug(packet['designation'])}"


def _trim(packet: dict, step: dict) -> dict:
    p = copy.deepcopy(packet)
    p["questions"] = p["questions"][: step["questions"]]
    p["topics"] = p["topics"][: step["topics"]]
    p["resume_focus"]["items"] = p["resume_focus"]["items"][: step["resume"]]
    p["role_focus"]["items"] = p["role_focus"]["items"][: step["role"]]
    p["trend"]["items"] = p["trend"]["items"][: step["trend"]]
    p["revision_priority"] = p["revision_priority"][: step["priority"]]
    return p


def build_pdf(packet: dict, pdf_path: Path, png_path: Path) -> tuple[dict, list[dict], list[str]]:
    """Render, check, and re-render with trims until every check passes (or trims run out)."""
    fixes: list[str] = []
    candidates = [packet] + [_trim(packet, st) for st in TRIM_STEPS]
    best = None
    for i, pk in enumerate(candidates):
        layout = render(pk, str(pdf_path))
        checks = run_checks(str(pdf_path), str(png_path), pk, layout)
        if all(c["passed"] for c in checks):
            if i:
                fixes.append(f"Layout fix applied: lists trimmed to lowest-priority limits (step {i}) "
                             "so the packet fits one readable page.")
            return pk, checks, fixes
        failed = [c for c in checks if not c["passed"]]
        if best is None or len(failed) < best[2]:
            best = (pk, checks, len(failed))
        only_fit_issues = all(c["check"] in ("Nothing clipped / all content inside margins", "Exactly one page",
                                             "Readable text size") for c in failed)
        if not only_fit_issues:
            break  # trimming will not fix content problems
    pk = best[0]
    layout = render(pk, str(pdf_path))
    checks = run_checks(str(pdf_path), str(png_path), pk, layout)
    fixes.append("Some quality checks still failed after automatic fixes; see the QC list.")
    return pk, checks, fixes


def evidence_csv(packet: dict) -> str:
    ids = packet["interview_ids"]
    buf = io.StringIO()
    w = csv.writer(buf)
    if packet.get("mode") == "role":
        w.writerow(["Note", "No historical interview transcripts supplied; no candidate-level occurrence data exists."])
        w.writerow([])
        w.writerow(["Expected topic", "Priority (stars)", "Basis", "Occurrence"])
        for t in packet["topics"]:
            w.writerow([t["name"], t["stars"], t["basis"], "not observed"])
        w.writerow([])
        w.writerow(["Likely question", "Priority (stars)", "Why", "Occurrence"])
        for q in packet["questions"]:
            w.writerow([q["text"], q["stars"], q["why"], "not observed"])
        return buf.getvalue()
    w.writerow(["Topic"] + [f"Candidate {i + 1}" for i in range(len(ids))] + ["Occurrence"])
    for t in packet["topics"]:
        w.writerow([t["name"]] + ["Yes" if iid in t["ids"] else "No" for iid in ids]
                   + [f"{t['count']}/{packet['n']}"])
    w.writerow([])
    w.writerow(["Question pattern"] + [f"Candidate {i + 1}" for i in range(len(ids))] + ["Occurrence"])
    for q in packet["questions"]:
        w.writerow([q["text"]] + ["Yes" if iid in q["ids"] else "No" for iid in ids]
                   + [f"{q['count']}/{packet['n']}"])
    return buf.getvalue()


def _role_methodology(packet: dict, checks: list[dict], fixes: list[str]) -> str:
    m = packet["meta"]
    L = [
        "INTERVIEW INTELLIGENCE PACKET - METHODOLOGY AND NOTES",
        f"Company: {packet['company']}",
        f"Designation: {packet['designation']} (as entered by the user)",
        f"Target round: {packet['target_round']}",
        f"Round info: {packet.get('round_info') or 'not supplied'}",
        f"Round type used: {(packet.get('round_type') or {}).get('label') or 'not determined'} "
        f"(source: {(packet.get('round_type') or {}).get('source', 'n/a')})",
        f"Generated: {m['generated']}   Model: {m.get('model', 'unknown')}",
        "Rules source: prompt.txt (sent unmodified), plus role-based mode instructions",
        "",
        "MODE: ROLE-BASED (no historical interview transcripts supplied)",
        "Number of transcript files supplied: 0",
        "Number of unique candidates / interview rounds: 0",
        "Frequency denominator used: none - no interviews were analyzed, so no counts,",
        "  frequencies or percentages appear anywhere in the packet.",
        f"Packet based on: {packet['source_label']}.",
        f"JD supplied: {'Yes - ' + m['jd_file'] if m.get('jd_file') else 'No'}",
        f"Resume supplied: {'Yes - ' + m['resume_file'] if m.get('resume_file') else 'No'}",
        "Chronological trend analysis possible: No (no interview data).",
        "",
        "All topics, questions and priorities are expectations for this designation and round,",
        "not observations. Upload real interview transcripts for evidence-based counts.",
    ]
    if m.get("rejected_files"):
        L.append("Files that could not be read:")
        L += [f"  {n}: {r}" for n, r in m["rejected_files"]]
    L += ["", "Expected topics and basis:"]
    L += [f"  [{t['stars']}/5] {t['name']} - {t['basis']}" for t in packet["topics"]]
    if m.get("notes"):
        L += ["", "Notes and caveats:"] + [f"  - {x}" for x in m["notes"]]
    L += ["", "Quality checks on the final PDF:"]
    L += [f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']} - {c['detail']}" for c in checks]
    L += [f"  {f}" for f in fixes]
    return "\n".join(L) + "\n"


def methodology(packet: dict, checks: list[dict], fixes: list[str]) -> str:
    if packet.get("mode") == "role":
        return _role_methodology(packet, checks, fixes)
    m, n = packet["meta"], packet["n"]
    L = []
    add = L.append
    add(f"INTERVIEW INTELLIGENCE PACKET - METHODOLOGY AND NOTES")
    add(f"Company: {packet['company']}")
    add(f"Designation: {packet['designation']} (as entered by the user)")
    add(f"Target round: {packet['target_round']} (secondary hint when transcripts are supplied; not shown on the PDF)")
    add(f"Round info: {packet.get('round_info') or 'not supplied'}")
    rt = packet.get("round_type") or {}
    add(f"Round type used: {rt.get('label') or 'not determined'} (source: {rt.get('source', 'n/a')})")
    add("Priority with transcripts: transcripts > round info > JD/resume > round number")
    add(f"Generated: {m['generated']}   Model: {m.get('model', 'unknown')}")
    add("Rules source: prompt.txt (sent unmodified with every model call)")
    add("")
    add(f"Number of transcript files supplied: {len(m['transcript_files'])}")
    for fid, name in m["transcript_files"]:
        add(f"  {fid}: {name}")
    if m.get("rejected_files"):
        add("Files that could not be read (not analyzed):")
        for name, reason in m["rejected_files"]:
            add(f"  {name}: {reason}")
    add(f"Number of unique candidates: {len({g['candidate'].lower() for g in m['groups']})}")
    add(f"Number of unique interview rounds (candidate interviews): {n}")
    add(f"Grouping method: {m['grouping_source']}")
    if m.get("analysis_mode"):
        add(f"Analysis method: {m['analysis_mode']}")
    add("Transcript files merged per candidate interview:")
    for i, g in enumerate(m["groups"], 1):
        rnd = f", {g['round']}" if g["round"] else ""
        add(f"  Candidate {i} ({g['id']}, {g['candidate']}{rnd}): {' + '.join(g['files'])}")
        if g.get("reason"):
            add(f"      reason: {g['reason']}")
    add("Exact duplicates removed:")
    if m["duplicates"]:
        for dup, kept in m["duplicates"]:
            add(f"  {dup} (identical to {kept})")
    else:
        add("  none")
    add(f"Frequency denominator used: {n} unique candidate interviews (not the {len(m['transcript_files'])} files)")
    add("Counting rule: a topic or question counts once per candidate interview, however often it "
        "appears across that candidate's transcript parts. Counts are computed in code from the "
        "per-interview evidence listed in Evidence_Matrix.csv.")
    add("")
    add(f"JD supplied: {'Yes - ' + m['jd_file'] if m.get('jd_file') else 'No (section shown as ' + packet['role_focus']['title'] + ')'}")
    add(f"Resume supplied: {'Yes - ' + m['resume_file'] if m.get('resume_file') else 'No (section shown as ' + packet['resume_focus']['title'] + ')'}")
    add(f"Interview dates supplied: {'Yes' if m['dates_supplied'] else 'No'}; "
        f"interviews with dates mentioned in transcripts: {m['transcript_date_mentions']}")
    add(f"Chronological trend analysis possible: {'Yes' if packet['trend']['chronological'] else 'No'}"
        f" - section shown as '{packet['trend']['title']}'.")
    if m.get("trend_reason"):
        add(f"  Model reasoning: {m['trend_reason']}")
    add("")
    add("Topic evidence (one line per interview where the topic appeared):")
    for t in packet["topics"]:
        add(f"  {t['name']} - {t['count']}/{n}")
        for iid in t["ids"]:
            add(f"      {iid}: {t['evidence'].get(iid, '')}")
    add("")
    add("Question evidence:")
    for q in packet["questions"]:
        add(f"  [{q['stars']}/5] {q['text']} - {q['count']}/{n} ({', '.join(q['ids'])})")
        if q["why"]:
            add(f"      why prioritized: {q['why']}")
    if m.get("notes"):
        add("")
        add("Notes and caveats:")
        for note in m["notes"]:
            add(f"  - {note}")
    add("")
    add("Quality checks on the final PDF:")
    for c in checks:
        add(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']} - {c['detail']}")
    for f in fixes:
        add(f"  {f}")
    return "\n".join(L) + "\n"


def build_all(packet: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base_name(packet)
    pdf_name = f"{base}_Interview_Intelligence_Packet.pdf"
    png_name = f"{base}_Interview_Intelligence_Packet_Preview.png"
    zip_name = f"{base}_Interview_Intelligence_Package.zip"
    pdf_path, png_path = out_dir / pdf_name, out_dir / png_name

    final_packet, checks, fixes = build_pdf(packet, pdf_path, png_path)
    csv_text = evidence_csv(final_packet)
    notes_text = methodology(final_packet, checks, fixes)
    with zipfile.ZipFile(out_dir / zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pdf_path, pdf_name)
        z.write(png_path, png_name)
        z.writestr("Methodology_and_Notes.txt", notes_text)
        z.writestr("Evidence_Matrix.csv", csv_text)
    return {"pdf": pdf_name, "png": png_name, "zip": zip_name, "checks": checks, "fixes": fixes}
