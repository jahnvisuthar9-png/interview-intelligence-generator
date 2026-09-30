"""Transcript grouping per prompt.txt "CRITICAL TRANSCRIPT-HANDLING RULE".

Steps implemented here (deterministic, before any model call):
  1. list all transcript files
  2. normalize candidate names from file names (Company + Candidate Name)
  3. group files by candidate (suffixes like -1 / -2 / (3) / part 2 are treated as parts)
  5. remove exact duplicate transcript exports (content hash after whitespace normalization)
The model later reviews the proposed grouping (it may see speaker names inside the files),
and the final grouping is validated in code so every file is used exactly once.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

NOISE_WORDS = {
    "interview", "interviews", "transcript", "transcripts", "recording", "recordings",
    "meeting", "meet", "zoom", "teams", "call", "video", "audio", "final", "copy",
    "export", "exported", "captions", "caption", "subtitles", "vtt", "txt", "pdf", "docx",
    "part", "pt", "file", "session", "with", "and", "the", "for", "of", "ai", "ml",
    "round", "r1", "r2", "r3", "r4", "screen", "screening", "technical", "hr",
}


@dataclass
class TranscriptFile:
    file_id: str
    filename: str
    text: str
    content_hash: str
    part_number: int | None = None
    proposed_candidate: str = ""
    duplicate_of: str | None = None  # file_id of the kept copy, if an exact duplicate


@dataclass
class InterviewGroup:
    interview_id: str          # "C1", "C2", ...
    candidate_label: str       # normalized candidate name, e.g. "Sohan"
    round_label: str           # "" unless the data proves a separate round
    file_ids: list[str] = field(default_factory=list)
    merged_text: str = ""


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def _strip_extension(name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name)


def parse_filename(filename: str, company: str) -> tuple[str, int | None]:
    """Return (normalized candidate name, part number) from a transcript file name."""
    stem = _strip_extension(filename)
    part = None
    # trailing "(2)", "-2", "_2", " 2", "part 2", "pt2"
    m = re.search(r"(?:[\s_\-.]*(?:part|pt)?[\s_\-.]*\(?(\d{1,2})\)?)\s*$", stem, re.I)
    if m and m.group(1) and m.start() > 0:
        part = int(m.group(1))
        stem = stem[: m.start()]
    # drop dates like 2026-05-01, 01-05-2026, 20260501
    stem = re.sub(r"\b\d{4}[-_.]\d{1,2}[-_.]\d{1,2}\b|\b\d{1,2}[-_.]\d{1,2}[-_.]\d{2,4}\b|\b\d{8}\b", " ", stem)
    tokens = [t for t in re.split(r"[\s_\-.,()\[\]]+", stem) if t]
    company_tokens = {t.lower() for t in re.split(r"[\s_\-.,]+", company or "") if t}
    kept = []
    for t in tokens:
        tl = t.lower()
        if tl in company_tokens or tl in NOISE_WORDS or re.fullmatch(r"\d+", tl):
            continue
        kept.append(t)
    name = " ".join(w[:1].upper() + w[1:].lower() for w in kept)
    return name, part


def build_files(raw: list[tuple[str, str]], company: str) -> list[TranscriptFile]:
    """raw = [(filename, extracted_text)] -> TranscriptFile list with duplicates flagged."""
    files: list[TranscriptFile] = []
    first_by_hash: dict[str, str] = {}
    for i, (filename, text) in enumerate(raw, start=1):
        fid = f"F{i}"
        h = content_hash(text)
        name, part = parse_filename(filename, company)
        tf = TranscriptFile(fid, filename, text, h, part, name)
        if h in first_by_hash:
            tf.duplicate_of = first_by_hash[h]
        else:
            first_by_hash[h] = fid
        files.append(tf)
    return files


def propose_groups(files: list[TranscriptFile]) -> list[dict]:
    """Deterministic proposal: group non-duplicate files by normalized candidate name.
    Files whose name could not be parsed become their own group (flagged for model review)."""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for f in files:
        if f.duplicate_of:
            continue
        key = f.proposed_candidate.lower() if f.proposed_candidate else f"__unnamed_{f.file_id}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(f.file_id)
    proposal = []
    for key in order:
        first = next(f for f in files if f.file_id == groups[key][0])
        proposal.append({
            "candidate_label": first.proposed_candidate or "UNKNOWN (name not found in file name)",
            "file_ids": groups[key],
        })
    return proposal


def finalize_groups(files: list[TranscriptFile], groups: list[dict]) -> list[InterviewGroup]:
    """Validate a grouping (from the model or the proposal) and merge parts.

    Guarantees: each non-duplicate file appears in exactly one group; duplicates are excluded;
    parts are merged in part-number order; identical parts inside a group are dropped once more.
    """
    by_id = {f.file_id: f for f in files}
    usable = [f.file_id for f in files if not f.duplicate_of]
    seen: set[str] = set()
    clean: list[dict] = []
    for g in groups:
        ids = []
        for fid in g.get("file_ids", []):
            if fid in by_id and not by_id[fid].duplicate_of and fid not in seen:
                ids.append(fid)
                seen.add(fid)
        if ids:
            clean.append({
                "candidate_label": (g.get("candidate_label") or "").strip() or "Unknown",
                "round_label": (g.get("round_label") or "").strip(),
                "file_ids": ids,
            })
    for fid in usable:  # anything the grouping forgot becomes its own interview
        if fid not in seen:
            f = by_id[fid]
            clean.append({"candidate_label": f.proposed_candidate or "Unknown",
                          "round_label": "", "file_ids": [fid]})

    result: list[InterviewGroup] = []
    for i, g in enumerate(clean, start=1):
        ids = sorted(g["file_ids"], key=lambda x: (by_id[x].part_number or 0, by_id[x].filename.lower()))
        texts, hashes = [], set()
        for fid in ids:
            f = by_id[fid]
            if f.content_hash in hashes:
                continue
            hashes.add(f.content_hash)
            texts.append(f"--- transcript part: {f.filename} ---\n{f.text}")
        result.append(InterviewGroup(f"C{i}", g["candidate_label"], g["round_label"], ids, "\n\n".join(texts)))
    return result
