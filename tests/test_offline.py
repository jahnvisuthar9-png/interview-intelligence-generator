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

    def __init__(self):
        self.calls = []

    def structured(self, system, user, schema, name):
        self.calls.append(name)
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
        if name == "role_based_packet":
            assert "ROLE-BASED MODE" in system and "NOT PROVIDED" in user
            assert "YOUR OWN KNOWLEDGE" in system and "Use only the supplied material" not in system
            return ROLE_RESPONSE
        payload = json.loads(user)
        groups = None
        if name == "interview_intelligence_grouped":  # grouping + analysis in one call
            assert "STEP 1 - GROUPING" in system and "TARGET ROUND" in system
            assert all(f["text"] for f in payload["files"])
            groups = [{"interview_id": f"C{i}", "candidate_label": g["candidate_label"], "round_label": "",
                       "file_ids": g["file_ids"], "reason": "test"}
                      for i, g in enumerate(payload["deterministic_proposal"], 1)]
            payload["interview_ids"] = [g["interview_id"] for g in groups]
        if "interview_rounds" in payload:  # aggregate stage: every field + round info must be present
            assert "TARGET ROUND" in system and "Use EVERY supplied input" in system
            evidence_key = "interviews" if name.endswith("single_pass") else "per_interview_extractions"
            for k in ("company", "designation", "target_round", "interview_dates", "job_description",
                      "candidate_resume", "interview_rounds", evidence_key):
                assert k in payload, k
            if name.endswith("single_pass"):
                assert all(i["merged_transcript"] for i in payload["interviews"])
                assert "check EVERY interview" in system
            assert all("file_names" in r for r in payload["interview_rounds"])
        ids = payload["interview_ids"]
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
            "interviews": [{"interview_id": i, "contains_real_interview_content": True,
                            "round_evidence": "Round 2", "date_evidence": ""} for i in ids],
            **({"groups": groups} if groups else {}),
        }


ROLE_RESPONSE = {
    "topics": [{"name": n, "stars": st, "basis": "role"} for n, st in [
        ("Current ML project deep dive and ownership", 5), ("Production ML: deploy, monitor, debug", 5),
        ("LLM / RAG system design", 4), ("Model evaluation and metrics", 4),
        ("Coding / DSA fundamentals", 4), ("Behavioral: collaboration, ownership", 3)]],
    "top_questions": [{"question": q, "stars": st, "why_prioritized": "role"} for q, st in [
        ("Walk me through your most recent ML project end to end.", 5),
        ("How would you take this model from notebook to production?", 5),
        ("Design a RAG system for internal documents.", 4),
        ("How do you choose evaluation metrics for an imbalanced problem?", 4),
        ("Tell me about a time a model failed in production.", 3)]],
    "resume_focus": [{"area": a, "stars": st, "basis": "role"} for a, st in [
        ("Exact personal contribution on recent projects", 5), ("Production deployment experience", 5),
        ("Model and architecture trade-offs", 4), ("Measured business impact", 4)]],
    "role_focus": [{"area": a, "stars": st, "basis": "role"} for a, st in [
        ("Production ML", 5), ("LLM / GenAI", 5), ("Model evaluation", 4), ("Python and data", 4)]],
    "signal": [{"topic": t, "direction": d} for t, d in [
        ("Production ML", "up"), ("LLM / RAG design", "up"), ("Model evaluation", "stable"),
        ("Classical ML theory", "down")]],
    "revision_priority": [{"item": i, "stars": st} for i, st in [
        ("Project story with exact ownership", 5), ("Production ML scenario", 5),
        ("RAG design trade-offs", 4), ("Evaluation metrics", 4)]],
    "revision_order": [{"minutes": m, "item": i} for m, i in [
        (10, "Project story with exact ownership"), (8, "Production ML scenario"),
        (7, "RAG design trade-offs"), (5, "Evaluation metrics")]],
    "tomorrow": {"review": [{"item": "Project story with ownership", "stars": 5},
                            {"item": "Production ML lifecycle", "stars": 5}],
                 "practice_out_loud": [{"item": "90-second project walkthrough", "stars": 5},
                                       {"item": "RAG design out loud", "stars": 4}],
                 "remember": ["State your exact contribution", "Explain trade-offs, not just choices"]},
    "caveats": ["No transcripts supplied."],
}


def slow_paths():
    """Large batch -> per-interview path; unnamed file -> grouping review. Same counts either way."""
    import os
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Interviewer>Explain RAG {}</v>\n"
    raw = [(f"Google-{n}-{p}.vtt", extract_text("x.vtt", vtt.format(n + str(p)).encode()))
           for n in NAMES[:4] for p in (1, 2)]
    os.environ["SINGLE_PASS_MAX_CHARS"] = "0"
    llm = FakeLLM()
    big = run(Inputs("Google", "AI/ML Engineer", "Round 2", raw), llm)
    os.environ.pop("SINGLE_PASS_MAX_CHARS")
    print("model calls (large batch):", len(llm.calls))
    assert llm.calls.count("interview_extraction") == 4 and big["n"] == 4
    llm2 = FakeLLM()
    unnamed = run(Inputs("Google", "AI/ML Engineer", "Round 2", raw + [("transcript.txt", "Interviewer: hi")]), llm2)
    print("model calls (unnamed file):", llm2.calls)
    assert llm2.calls == ["interview_intelligence_grouped"] and unnamed["n"] == 5
    assert unnamed["topics"][0]["count"] == 5  # model ids remapped onto validated ids
    os.environ["SINGLE_PASS_MAX_CHARS"] = "0"
    llm3 = FakeLLM()
    run(Inputs("Google", "AI/ML Engineer", "Round 2", raw + [("transcript.txt", "Interviewer: hi")]), llm3)
    os.environ.pop("SINGLE_PASS_MAX_CHARS")
    print("model calls (unnamed + large batch):", len(llm3.calls))
    assert llm3.calls[0] == "transcript_grouping"
    fast = run(Inputs("Google", "AI/ML Engineer", "Round 2", raw), FakeLLM())
    assert [t["count"] for t in fast["topics"]] == [t["count"] for t in big["topics"]]
    # repeat request: identical inputs -> 0 calls; any change -> 1 call
    import tempfile
    cache = tempfile.mkdtemp()
    inp = Inputs("Google", "AI/ML Engineer", "Round 2", raw)
    first, again, changed = FakeLLM(), FakeLLM(), FakeLLM()
    p1 = run(inp, first, cache_dir=cache)
    p2 = run(inp, again, cache_dir=cache)
    run(Inputs("Google", "AI/ML Engineer", "Round 3", raw), changed, cache_dir=cache)
    print("model calls (first / identical repeat / changed round):", len(first.calls), len(again.calls), len(changed.calls))
    assert (len(first.calls), len(again.calls), len(changed.calls)) == (1, 0, 1)
    assert [t["count"] for t in p1["topics"]] == [t["count"] for t in p2["topics"]]
    build_all(p2, ROOT / "tests" / "out" / "cached")  # cached packet renders normally


def role_mode():
    cases = [
        ("role_only", Inputs("Google", "AI/ML Engineer", "Round 2", [])),
        ("jd_text", Inputs("Google", "AI/ML Engineer", "Round 2", [], jd=("pasted text", "Build RAG systems."))),
        ("jd_resume", Inputs("Google", "AI/ML Engineer", "Round 3", [], jd=("jd.pdf + pasted text", "JD"),
                             resume=("cv.pdf", "Resume"))),
    ]
    for label, inp in cases:
        packet = run(inp, FakeLLM())
        assert packet["mode"] == "role" and packet["n"] == 0
        out = build_all(packet, ROOT / "tests" / "out" / label)
        failed = [c for c in out["checks"] if not c["passed"]]
        print(label, packet["role_focus"]["title"], "|", packet["resume_focus"]["title"],
              "| checks:", "all passed" if not failed else failed)
        assert not failed


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
    llm = FakeLLM()
    packet = run(inp, llm, progress=print)
    print("model calls (fast path):", len(llm.calls), llm.calls)
    assert llm.calls == ["interview_intelligence_single_pass"]
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
    slow_paths()
    role_mode()
