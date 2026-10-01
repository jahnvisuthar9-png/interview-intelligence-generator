"""Quota / fallback behaviour of the API client, using the real openai error classes.
Run: python -m tests.test_llm_fallback"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openai  # noqa: E402

try:
    import httpx2 as httpx  # newer openai releases
except ImportError:  # pragma: no cover
    import httpx

from pipeline import llm as L  # noqa: E402

DAILY = ("Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota. "
         "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
         "limit: 20, model: gemini-3.8-flash. Please retry in 51.26s.', 'status': 'RESOURCE_EXHAUSTED', "
         "'details': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}}]")
MINUTE = ("Error code: 429 - Quota exceeded: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. "
          "Please retry in 1.5s.")


def err(cls, code, msg):
    resp = httpx.Response(code, request=httpx.Request("POST", "https://example.test"))
    return cls(msg, response=resp, body=None)


def ok(payload):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload), refusal=None))])


def client_with(script, models="gemini-3.8-flash", fallbacks="gemini-3.5-flash-lite,gemini-3.1-flash-lite"):
    os.environ.update(OPENAI_API_KEY="test", OPENAI_MODEL=models, OPENAI_FALLBACK_MODELS=fallbacks,
                      OPENAI_MAX_RETRIES="2")
    c = L.OpenAIClient()
    calls = []

    def fake_call(model, *a, **k):
        calls.append(model)
        outcome = script(model, len(calls))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    c._call = fake_call
    return c, calls


def main():
    # 1. Your exact error: daily quota on 3.8 Flash -> no waiting, switches to Flash-Lite at once.
    c, calls = client_with(lambda m, n: err(openai.RateLimitError, 429, DAILY) if m == "gemini-3.8-flash" else ok({"x": 1}))
    t = time.time()
    assert c.structured("s", "u", {}, "role_based_packet") == {"x": 1}
    assert calls == ["gemini-3.8-flash", "gemini-3.5-flash-lite"] and time.time() - t < 1
    # a second request in the same job skips the exhausted model entirely
    c.structured("s", "u", {}, "again")
    assert calls[-1] == "gemini-3.5-flash-lite" and calls.count("gemini-3.8-flash") == 1
    print("daily limit -> instant fallback:", calls, "| used:", c.used_label, "| note:", c.events[0])

    # 2. Per-minute limit -> waits the requested 1.5 s, retries the SAME model, succeeds.
    c, calls = client_with(lambda m, n: err(openai.RateLimitError, 429, MINUTE) if n == 1 else ok({"y": 2}))
    t = time.time()
    assert c.structured("s", "u", {}, "p") == {"y": 2} and calls == ["gemini-3.8-flash"] * 2
    print(f"per-minute limit -> waited {time.time() - t:.1f}s, same model retried")

    # 3. Wrong model name -> next model.
    c, calls = client_with(lambda m, n: err(openai.NotFoundError, 404, "models/x is not found") if m == "gemini-3.5-flash-lite" else
                           err(openai.RateLimitError, 429, DAILY) if m == "gemini-3.8-flash" else ok({"z": 3}))
    assert c.structured("s", "u", {}, "p") == {"z": 3}
    print("unknown model name -> next fallback:", calls)

    # 4. Everything exhausted -> one clear message, no raw JSON dump.
    c, _ = client_with(lambda m, n: err(openai.RateLimitError, 429, DAILY))
    try:
        c.structured("s", "u", {}, "p")
        raise AssertionError("should fail")
    except L.LLMError as e:
        msg = str(e)
        assert "daily limit" in msg and "midnight Pacific" in msg and "RESOURCE_EXHAUSTED" not in msg
        print("all exhausted ->", msg)

    # 5. Bad key -> plain instruction.
    c, _ = client_with(lambda m, n: err(openai.AuthenticationError, 401, "API key not valid"))
    try:
        c.structured("s", "u", {}, "p")
    except L.LLMError as e:
        assert "API key was rejected" in str(e)
        print("bad key ->", e)
    # 6. Your exact 400 "Unsupported field: response_format" -> same model retried without it.
    os.environ.update(OPENAI_API_KEY="test", OPENAI_MODEL="some-new-model", OPENAI_FALLBACK_MODELS="")
    c = L.OpenAIClient()
    sent = []

    def create(**kw):
        sent.append(kw)
        if "response_format" in kw:
            raise err(openai.BadRequestError, 400, "Error code: 400 - {'error': {'message': 'Unsupported field: "
                      "response_format.', 'type': 'invalid_request_error', 'param': 'response_format'}}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='```json\n{"ok": true}\n```', refusal=None))])

    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert c.structured("SYS", "u", {"type": "object"}, "role_based_packet") == {"ok": True}
    assert "response_format" in sent[0] and "response_format" not in sent[1]
    assert "OUTPUT FORMAT" in sent[1]["messages"][0]["content"] and sent[1]["model"] == "some-new-model"
    c.structured("SYS", "u", {"type": "object"}, "second")  # remembered: no rejected attempt again
    assert len(sent) == 3 and "response_format" not in sent[2]
    print("response_format rejected -> retried without it, fenced JSON parsed:", c.events[0])
    print("ALL FALLBACK TESTS PASSED")


if __name__ == "__main__":
    main()
