"""Offline end-to-end test: no OpenAI key needed. Run:  python -m tests.test_offline
A fake LLM returns fixed structured output so grouping, counting, rendering, QC and packaging
can be verified. (The real app always uses OpenAI.)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.analysis import Inputs, run  # noqa: E402
from pipeline.extract import extract_text  # noqa: E402
from pipeline.package import build_all  # noqa: E402

NAMES = ["Sohan", "Goutham", "Priya", "Arjun", "Meera", "Kiran", "Ravi", "Anita", "Dev", "Isha"]


class FakeLLM:
    model = "fake-offline"

    def structured(self, system, user, schema, name):
        assert "CRITICAL TRANSCRIPT-HANDLING RULE" in system  # prompt.txt is always sent
        if name == "transcript_grouping":
            payload = json.loads(user)
            return {"groups": [dict(g, round_label="", reason="same candidate name, part suffixes")
                               for g in payload["deterministic_proposal"]], "notes": []}
        if name == "interview_extraction":
            return {"interview_id": "", "contains_real_interview_content": True,
                    "questions": [{"pattern": "Explain your current AI/ML project end to end.",
                                   "topic": "Project", "kind": "project_deep_dive", "follow_up_count": 3,
                                   "depth": "high", "evidence": "walk me through your project"}],
                    "experience_probes": ["exact contribution"], "round_evidence": "", "date_evidence": ""}
        ids = json.loads(user)["interview_ids"]
        occ = lambda k: [{"interview_id": i, "evidence": "asked"} for i in ids[:k]]
        return {
            "topics": [
                {"name": "Coding / DSA (algorithm problem, complexity analysis)", "occurrences": occ(7)},
                {"name": "Production ML reasoning (deploy, debug, scale a live system)", "occurrences": occ(7)},
                {"name": "LLM / GenAI reasoning (RAG, fine-tuning, inference, agents)", "occurrences": occ(6)},
                {"name": "Current AI/ML project deep dive, exact contribution", "occurrences": occ(5)},
                {"name": "RAG / retrieval / vector search", "occurrences": occ(4)},
                {"name": "End-to-end ML system design", "occurrences": occ(4)},
                {"name": "Model evaluation / monitoring / drift", "occurrences": occ(4)},
                {"name": "Behavioral / Googliness round", "occurrences": occ(2) + [{"interview_id": "C99", "evidence": "x"}]},
            ],
            "top_questions": [
                {"question": "Explain your current AI/ML project end to end and your exact contribution.", "stars": 5, "interview_ids": ids[:5], "why_prioritized": "many follow-ups"},
                {"question": "Design or extend an ML/LLM system in production, from ingestion to monitoring.", "stars": 5, "interview_ids": ids[:4], "why_prioritized": ""},
                {"question": "Solve a coding/DSA problem and explain time and space complexity.", "stars": 5, "interview_ids": ids[:7], "why_prioritized": ""},
                {"question": "Explain your RAG or retrieval design and the trade-offs you made.", "stars": 4, "interview_ids": ids[:4], "why_prioritized": ""},
                {"question": "How would you debug or roll back a production ML failure?", "stars": 4, "interview_ids": ids[:3], "why_prioritized": ""},
                {"question": "Why this model or architecture, and what trade-offs did you weigh?", "stars": 4, "interview_ids": ids[:3], "why_prioritized": ""},
                {"question": "Tell me about a time you had more work than you could manage.", "stars": 3, "interview_ids": ids[:2], "why_prioritized": ""},
                {"question": "Invented question with no evidence", "stars": 5, "interview_ids": [], "why_prioritized": ""},
            ],
            "resume_focus": [
                {"area": "Current AI/ML project and exact personal ownership", "stars": 5, "basis": ""},
                {"area": "Production deployment and measured impact, not just accuracy", "stars": 5, "basis": ""},
                {"area": "Architecture/model choices and the trade-offs behind them", "stars": 4, "basis": ""},
                {"area": "Coding fluency independent of the AI/ML narrative", "stars": 4, "basis": ""},
            ],
            "role_focus": [
                {"area": "Production ML reasoning", "stars": 5, "basis": ""},
                {"area": "Coding / DSA fundamentals", "stars": 5, "basis": ""},
                {"area": "LLM / GenAI, RAG, agents", "stars": 4, "basis": ""},
                {"area": "System design at scale", "stars": 4, "basis": ""},
                {"area": "Behavioral / cross-team collaboration", "stars": 3, "basis": ""},
            ],
            "trend": {"chronological_analysis_possible": True, "reason": "claims dates",
                      "items": [{"topic": "Coding / DSA", "direction": "up"},
                                {"topic": "Production ML reasoning", "direction": "up"},
                                {"topic": "LLM / GenAI reasoning", "direction": "stable"},
                                {"topic": "Current project deep dive", "direction": "stable"},
                                {"topic": "Behavioral round", "direction": "down"}]},
            "revision_priority": [
                {"item": "Project story: problem, design, your contribution, impact", "stars": 5},
                {"item": "Coding/DSA warm-up: complexity, edge cases, clean code", "stars": 5},
                {"item": "Production ML: debug, rollback, scale scenario", "stars": 4},
                {"item": "LLM/RAG design trade-offs", "stars": 3},
                {"item": "Googliness / behavioral answers ready", "stars": 2},
            ],
            "revision_order": [
                {"minutes": 10, "item": "Project story: problem, design, your contribution, impact"},
                {"minutes": 7, "item": "Coding/DSA warm-up: complexity, edge cases, clean code"},
                {"minutes": 6, "item": "Production ML: debug, rollback, scale scenario"},
                {"minutes": 5, "item": "LLM/RAG design trade-offs"},
                {"minutes": 3, "item": "Googliness / behavioral answers ready"},
            ],
            "tomorrow": {
                "review": [{"item": "Project story with exact ownership", "stars": 5},
                           {"item": "One production-failure or scale scenario", "stars": 5},
                           {"item": "RAG or fine-tuning trade-offs", "stars": 4}],
                "practice_out_loud": [{"item": "90-second project walkthrough", "stars": 5},
                                      {"item": "A coding problem, narrating complexity", "stars": 5},
                                      {"item": "One Googliness-style behavioral story", "stars": 4}],
                "remember": ["State your exact contribution, not the team's",
                             "Clarify assumptions before coding", "Quantify impact wherever possible"],
            },
            "caveats": ["Test data only."],
        }


def main():
    raw = []
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\n<v Interviewer>Can you hear me?</v>\n\n2\n00:00:05.000 --> 00:00:09.000\n<v Interviewer>Walk me through your project {}.</v>\n"
    for i, nm in enumerate(NAMES):
        parts = 1 + (i % 3)
        for p in range(1, parts + 1):
            fname = f"Google-{nm}-{p}.vtt" if parts > 1 else f"Google {nm}.vtt"
            raw.append((fname, extract_text(fname, vtt.format(f"{nm}{p}").encode())))
    raw.append(("Google-Sohan-1 (copy).vtt", raw[0][1]))  # exact duplicate export
    inp = Inputs("Google", "AI/ML Engineer", "Round 2", raw)
    packet = run(inp, FakeLLM(), progress=print)
    assert packet["n"] == 10, packet["n"]
    assert not packet["trend"]["chronological"], "trend must not be chronological without dates"
    assert all(q["count"] > 0 for q in packet["questions"])
    assert packet["topics"][-1]["count"] == 2  # invalid C99 ignored
    out = build_all(packet, ROOT / "tests" / "out")
    for c in out["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"] + " - " + c["detail"])
    print(out["fixes"])
    print(f"files: {len(raw)} -> interviews: {packet['n']}")
    assert all(c["passed"] for c in out["checks"])


if __name__ == "__main__":
    main()
