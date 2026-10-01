"""OpenAI calls. The API key is read server-side from OPENAI_API_KEY and never sent to the browser.

Every stage sends the full, unmodified prompt.txt as the governing instructions, followed by a
short stage task that only explains which part of prompt.txt this call performs and the JSON shape.
"""
from __future__ import annotations

import json
import os
import re
import time
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
    "round_type": _obj({"label": S, "source": {"type": "string", "enum": [
        "round info", "transcripts", "round number", "company process"]}}),
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


# ---------- Single pass: sections A-G plus per-interview facts in one call ----------
SINGLE_PASS_SCHEMA = _obj(dict(ANALYSIS_SCHEMA["properties"], interviews=_arr(_obj({
    "interview_id": S,
    "contains_real_interview_content": B,
    "round_evidence": S,
    "date_evidence": S,
}))))

# ---------- One call that also groups files whose names have no candidate name ----------
GROUP_AND_ANALYZE_SCHEMA = _obj(dict(SINGLE_PASS_SCHEMA["properties"], groups=_arr(_obj({
    "interview_id": S,
    "candidate_label": S,
    "round_label": S,
    "file_ids": _arr(S),
    "reason": S,
}))))

# ---------- Role-based mode (no transcripts supplied) ----------
ROLE_SCHEMA = _obj({
    "round_type": _obj({"label": S, "source": {"type": "string", "enum": [
        "round info", "transcripts", "round number", "company process"]}}),
    "topics": _arr(_obj({"name": S, "stars": STARS, "basis": S})),
    "top_questions": _arr(_obj({"question": S, "stars": STARS, "why_prioritized": S})),
    "resume_focus": _arr(_obj({"area": S, "stars": STARS, "basis": S})),
    "role_focus": _arr(_obj({"area": S, "stars": STARS, "basis": S})),
    "signal": _arr(_obj({"topic": S, "direction": {"type": "string", "enum": ["up", "stable", "down"]}})),
    "revision_priority": _arr(_obj({"item": S, "stars": STARS})),
    "revision_order": _arr(_obj({"minutes": I, "item": S})),
    "tomorrow": _obj({
        "review": _arr(_obj({"item": S, "stars": STARS})),
        "practice_out_loud": _arr(_obj({"item": S, "stars": STARS})),
        "remember": _arr(S),
    }),
    "caveats": _arr(S),
})

ROLE_MODE_ADDENDUM = """
==================================================
ROLE-BASED MODE (no historical transcripts supplied)
==================================================
The user supplied NO historical interview transcripts. The transcript-handling, frequency and
candidate-count rules above therefore cannot be applied, and you must not pretend they were.
Instead, build the same packet sections from the other fields: company, designation, target round,
ROUND INFO (the user's description of the round, when supplied), and the JD and/or candidate resume
if present, combined with YOUR OWN KNOWLEDGE of how this
company interviews for this role at this round (its known process, round structure, question
style, focus areas and reported interview patterns for similar roles and rounds).
- Questions must be what this company realistically asks in THIS round for THIS role, not
  generic or random interview questions. If the round is an early/initial screen, reflect a
  screen; if it is a coding, system-design, hiring-manager, behavioral or final round, reflect
  that. If you know little about this company, use the closest well-known patterns for its
  size, industry and this role, and say so in the caveats.
- Tailor to the JD (required skills, responsibilities) and the resume (projects and claims most
  likely to be probed) when supplied.
- Follow every other rule in prompt.txt (no generic filler, stars as the priority indicator, JD
  Focus only if a JD is supplied, Resume Focus only if a resume is supplied, no chronological
  trend claims).
- Never output counts, percentages or "X of Y interviews"; nothing here was observed. In each
  "basis", say where it comes from (company interview pattern, round, JD line or resume item).
"""

APP_ADDENDUM = """
==================================================
APP INTEGRATION NOTES (added by the Interview Intelligence Generator app)
==================================================
The instructions above (prompt.txt) are the source of truth and override anything here.
This app splits prompt.txt into stages and renders the PDF, ZIP, CSV and methodology itself.
In this call you perform ONLY the stage described below and return JSON that matches the
supplied schema exactly. Never invent statistics, JD alignment, trends, dates or candidate
information. When transcripts are supplied, topics and questions must come from them.
"""


class LLMError(Exception):
    pass


def _quota_kind(err: Exception) -> str:
    """'daily' when a per-day quota is used up (retrying today is pointless), else 'minute'."""
    text = str(err)
    if re.search(r"PerDay|per[ _-]?day|RequestsPerDay|daily", text, re.I):
        return "daily"
    return "minute"


def _retry_seconds(err: Exception, default: float = 20.0) -> float:
    m = re.search(r"retry in ([0-9.]+)s|retryDelay['\"]?:\s*['\"]?([0-9.]+)s", str(err))
    val = float(m.group(1) or m.group(2)) if m else default
    return min(max(val, 1.0), 65.0)


def _short(err: Exception, limit: int = 300) -> str:
    msg = getattr(err, "message", None) or str(err)
    msg = re.sub(r"\s+", " ", str(msg))
    return msg[:limit] + ("…" if len(msg) > limit else "")


class OpenAIClient:
    """Calls any OpenAI-compatible API (OpenAI, Gemini, ...).

    - Per-minute rate limits: waits the time the provider asks for (max ~1 min), then retries.
    - Daily quota used up / model not available: moves on to the next model in
      OPENAI_FALLBACK_MODELS instead of retrying something that cannot succeed today.
    - Errors are turned into short, readable messages for the UI.
    """

    def __init__(self, model: str | None = None):
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise LLMError("No API key is set on the server (OPENAI_API_KEY).")
        from openai import OpenAI

        # Retries are handled below so that daily-quota errors fail fast instead of hanging.
        self.client = OpenAI(api_key=key, timeout=float(os.environ.get("OPENAI_TIMEOUT", "300")),
                             max_retries=0)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")
        fallbacks = [m.strip() for m in os.environ.get("OPENAI_FALLBACK_MODELS", "").split(",") if m.strip()]
        self.models = [self.model] + [m for m in fallbacks if m != self.model]
        self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
        self._exhausted: set[str] = set()   # models whose daily quota ran out during this job
        self._no_response_format: set[str] = set()  # models that reject the response_format field
        self.used_models: list[str] = []
        self.events: list[str] = []

    @property
    def used_label(self) -> str:
        return ", ".join(dict.fromkeys(self.used_models)) or self.model

    def _call(self, model: str, system: str, user: str, schema: dict, name: str):
        extra = {}
        effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
        if effort:  # optional speed knob; unset = model default (best quality)
            extra["reasoning_effort"] = effort
        if model in self._no_response_format:
            # This model/provider rejects response_format: give the exact JSON shape in the
            # instructions instead. The app validates and cleans the result as usual.
            system = (system + "\n\nOUTPUT FORMAT: reply with ONE JSON object only (no markdown, no "
                      "code fences, no text before or after) that matches this JSON Schema exactly, "
                      "with every listed property present:\n" + json.dumps(schema, ensure_ascii=False))
        else:
            extra["response_format"] = {"type": "json_schema",
                                        "json_schema": {"name": name, "strict": True, "schema": schema}}
        return self.client.chat.completions.create(
            **extra,
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )

    def structured(self, system: str, user: str, schema: dict, name: str) -> dict:
        import openai

        problems: list[str] = []
        for model in self.models:
            if model in self._exhausted:
                continue
            attempt = 0
            while True:
                try:
                    resp = self._call(model, system, user, schema, name)
                    break
                except openai.RateLimitError as e:
                    if _quota_kind(e) == "daily":
                        self._exhausted.add(model)
                        self.events.append(f"{model}: free daily limit reached; switched to the next model.")
                        problems.append(f"{model}: daily limit reached")
                        resp = None
                        break
                    attempt += 1
                    if attempt > self.max_retries:
                        problems.append(f"{model}: per-minute limit still reached after {self.max_retries} waits")
                        resp = None
                        break
                    time.sleep(_retry_seconds(e))
                except openai.NotFoundError as e:
                    self._exhausted.add(model)
                    self.events.append(f"{model}: model not available ({_short(e, 120)}).")
                    problems.append(f"{model}: model name not found")
                    resp = None
                    break
                except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise LLMError(f"The AI service did not respond ({name}): {_short(e)}") from e
                    time.sleep(min(2 ** attempt, 20))
                except openai.BadRequestError as e:
                    if "response_format" in str(e) and model not in self._no_response_format:
                        self._no_response_format.add(model)
                        self.events.append(f"{model}: does not accept response_format; the JSON shape was "
                                           "given in the instructions instead.")
                        continue  # same model, same request, without the unsupported field
                    raise LLMError(f"The AI service rejected the request ({name}): {_short(e)}") from e
                except openai.AuthenticationError as e:
                    raise LLMError("The API key was rejected. Check OPENAI_API_KEY in Render > "
                                   "Environment (copy it again with the copy button).") from e
                except openai.APIStatusError as e:
                    raise LLMError(f"The AI service returned an error ({name}): {_short(e)}") from e
                except Exception as e:
                    raise LLMError(f"AI request failed ({name}): {_short(e)}") from e
            if resp is not None:
                self.used_models.append(model)
                return self._parse(resp, name)

        if any("daily limit" in p for p in problems):
            raise LLMError(
                "The free daily limit is used up for: " + ", ".join(
                    p.split(":")[0] for p in problems if "daily limit" in p) +
                ". Google resets it at midnight Pacific time (about 12:30 PM India time). "
                "To keep going today, add another model to OPENAI_FALLBACK_MODELS or enable billing.")
        raise LLMError("No AI model could handle the request: " + "; ".join(problems))

    @staticmethod
    def _parse(resp, name: str) -> dict:
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"The model refused the {name} request: {choice.message.refusal}")
        content = (choice.message.content or "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Tolerate code fences or stray text around the JSON object.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"The model returned invalid JSON for {name}.")
