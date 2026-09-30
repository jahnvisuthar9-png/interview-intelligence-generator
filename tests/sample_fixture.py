"""Model output reproducing the attached sample PDF's content exactly (10 interviews).
Used to compare the renderer against the reference sample."""
IDS = [f"C{i}" for i in range(1, 11)]
occ = lambda k: [{"interview_id": i, "evidence": "from sample"} for i in IDS[:k]]
SAMPLE = {
    "topics": [
        {"name": "Coding / DSA (algorithm problem, complexity analysis)", "occurrences": occ(7)},
        {"name": "Production ML reasoning (deploy, debug, scale a live system)", "occurrences": occ(7)},
        {"name": "Current AI/ML project deep dive, own exact contribution", "occurrences": occ(5)},
        {"name": "LLM / GenAI reasoning (RAG, fine-tuning, inference, agents)", "occurrences": occ(6)},
        {"name": "RAG / retrieval / vector search", "occurrences": occ(4)},
        {"name": "End-to-end system design (not just a coding problem)", "occurrences": occ(4)},
        {"name": "Model evaluation / monitoring / drift", "occurrences": occ(4)},
        {"name": 'Behavioral / "Googliness" leadership round', "occurrences": occ(2)},
    ],
    "top_questions": [
        {"question": "Explain your current AI/ML project end to end and your exact contribution.", "stars": 5, "interview_ids": IDS[:5], "why_prioritized": ""},
        {"question": "Design or extend an ML/LLM system in production (ingestion through deployment and monitoring).", "stars": 5, "interview_ids": IDS[:4], "why_prioritized": ""},
        {"question": "Solve a coding/DSA problem and explain time/space complexity.", "stars": 5, "interview_ids": IDS[:7], "why_prioritized": ""},
        {"question": "Explain your RAG or retrieval design and the trade-offs you made.", "stars": 4, "interview_ids": IDS[:4], "why_prioritized": ""},
        {"question": "How would you debug or roll back a production ML failure?", "stars": 4, "interview_ids": IDS[:4], "why_prioritized": ""},
        {"question": "Why this model/architecture choice, and what trade-offs did you weigh?", "stars": 4, "interview_ids": IDS[:3], "why_prioritized": ""},
        {"question": "Tell me about a time you had more work than you could manage / got negative feedback (Googliness).", "stars": 3, "interview_ids": IDS[:2], "why_prioritized": ""},
    ],
    "resume_focus": [
        {"area": "Current AI/ML project and exact personal ownership", "stars": 5, "basis": ""},
        {"area": "Production deployment + measured impact, not just model accuracy", "stars": 5, "basis": ""},
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
    "trend": {"chronological_analysis_possible": False, "reason": "pool too small to claim a time trend",
              "items": [{"topic": "Coding / DSA (algorithm problem, complexity analysis)", "direction": "up"},
                        {"topic": "Production ML reasoning (deploy, debug, scale a live system)", "direction": "up"},
                        {"topic": "LLM / GenAI reasoning (RAG, fine-tuning, inference, agents)", "direction": "stable"},
                        {"topic": "Current AI/ML project deep dive, own exact contribution", "direction": "stable"},
                        {"topic": "RAG / retrieval / vector search", "direction": "stable"}]},
    "revision_priority": [
        {"item": "Project story: problem -> design -> your contribution -> impact", "stars": 5},
        {"item": "Coding/DSA warm-up (complexity, edge cases, clean code)", "stars": 5},
        {"item": "Production ML: debug/rollback/scale scenario", "stars": 4},
        {"item": "LLM/RAG design trade-offs", "stars": 3},
        {"item": "Googliness / behavioral answers ready", "stars": 2},
    ],
    "revision_order": [
        {"minutes": 10, "item": "Project story: problem -> design -> your contribution -> impact"},
        {"minutes": 7, "item": "Coding/DSA warm-up (complexity, edge cases, clean code)"},
        {"minutes": 6, "item": "Production ML: debug/rollback/scale scenario"},
        {"minutes": 5, "item": "LLM/RAG design trade-offs"},
        {"minutes": 2, "item": "Googliness / behavioral answers ready"},
    ],
    "tomorrow": {
        "review": [{"item": "Project story with exact ownership", "stars": 5},
                   {"item": "One production-failure or scale scenario", "stars": 5},
                   {"item": "RAG or fine-tuning trade-offs", "stars": 5}],
        "practice_out_loud": [{"item": "90-sec project walkthrough", "stars": 5},
                              {"item": "A coding problem out loud, narrating complexity", "stars": 5},
                              {"item": "One Googliness-style behavioral story", "stars": 5}],
        "remember": ["State your exact contribution, not the team's",
                     "Clarify assumptions before coding", "Quantify impact wherever possible"],
    },
    "caveats": [],
}
