"""OpenAI calls. The API key is read server-side from OPENAI_API_KEY and never sent to the browser.

Every stage sends the full, unmodified prompt.txt as the governing instructions, followed by a
short stage task that only explains which part of prompt.txt this call performs and the JSON shape.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt.txt"


def load_source_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": props, "required": list(props.keys())}


def _arr(items: dict) -> dict:
    return {"type": "array", "items": items}


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}

# ---------- Stage 1: grouping review ----------
GROUPING_SCHEMA = _obj({
    "groups": _arr(_obj({
        "candidate_label": S,
        "round_label": S,
        "file_ids": _arr(S),
        "reason": S,
    })),
    "notes": _arr(S),
})

# ---------- Stage 2: per-interview extraction ----------
EXTRACTION_SCHEMA = _obj({
    "interview_id": S,
    "contains_real_interview_content": B,
    "questions": _arr(_obj({
        "pattern": S,            # normalized, interview-ready wording
        "topic": S,              # short topic label
        "kind": {"type": "string", "enum": [
            "project_deep_dive", "technical_concept", "system_design", "coding",
            "scenario", "behavioral", "experience_probe", "other"]},
        "follow_up_count": I,
        "depth": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": S,           # short paraphrase of what the interviewer actually asked
    })),
    "experience_probes": _arr(S),  # which parts of the candidate's experience were challenged
    "round_evidence": S,           # any explicit round mention, else ""
    "date_evidence": S,            # any explicit date mention, else ""
})

# ---------- Stage 3: aggregate analysis ----------
STARS = {"type": "integer", "description": "1-5"}
EVIDENCE_ITEM = _obj({"interview_id": S, "evidence": S})

ANALYSIS_SCHEMA = _obj({
    "topics": _arr(_obj({"name": S, "occurrences": _arr(EVIDENCE_ITEM)})),
    "top_questions": _arr(_obj({"question": S, "stars": STARS,
                                "interview_ids": _arr(S), "why_prioritized": S})),
    "resume_focus": _arr(_obj({"area": S, "stars": STARS, "basis": S})),
    "role_focus": _arr(_obj({"area": S, "stars": STARS, "basis": S})),
    "trend": _obj({
        "chronological_analysis_possible": B,
        "reason": S,
        "items": _arr(_obj({"topic": S, "direction": {"type": "string", "enum": ["up", "stable", "down"]}})),
    }),
    "revision_priority": _arr(_obj({"item": S, "stars": STARS})),
    "revision_order": _arr(_obj({"minutes": I, "item": S})),
    "tomorrow": _obj({
        "review": _arr(_obj({"item": S, "stars": STARS})),
        "practice_out_loud": _arr(_obj({"item": S, "stars": STARS})),
        "remember": _arr(S),
    }),
    "caveats": _arr(S),
})


APP_ADDENDUM = """
==================================================
APP INTEGRATION NOTES (added by the Interview Intelligence Generator app)
==================================================
The instructions above (prompt.txt) are the source of truth and override anything here.
This app splits prompt.txt into stages and renders the PDF, ZIP, CSV and methodology itself.
In this call you perform ONLY the stage described below and return JSON that matches the
supplied schema exactly. Never invent statistics, questions, JD alignment, trends, dates,
or candidate information. Use only the supplied material.
"""


class LLMError(Exception):
    pass


class OpenAIClient:
    def __init__(self, model: str | None = None):
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise LLMError("OPENAI_API_KEY is not set on the server.")
        from openai import OpenAI

        self.client = OpenAI(api_key=key, timeout=float(os.environ.get("OPENAI_TIMEOUT", "300")),
                             max_retries=3)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")

    def structured(self, system: str, user: str, schema: dict, name: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": name, "strict": True, "schema": schema}},
            )
        except Exception as e:  # surface a readable message to the UI
            raise LLMError(f"OpenAI request failed ({name}): {e}") from e
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"OpenAI refused the {name} request: {choice.message.refusal}")
        content = choice.message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(f"OpenAI returned invalid JSON for {name}.") from e
