"""
Offline tests for ask_jev.py. No network, no pytest required.

    python test_ask_jev.py
"""

from __future__ import annotations

import asyncio
import json
import sys

from ask_jev import JevError, Tools

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def tool(**valves):
    t = Tools()
    t.valves.JEV_API_KEY = "test-key"
    for key, value in valves.items():
        setattr(t.valves, key, value)
    return t


def stub(tool_obj, responses):
    """Replace _send with a canned-response queue, recording every payload."""
    sent = []
    queue = list(responses)

    async def fake_send(session, payload, emitter):
        sent.append(payload)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    tool_obj._send = fake_send
    return sent


# ----------------------------------------------------------------------
# Fake aiohttp session, for exercising the real retry loop in _send
# ----------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, url, headers=None, json=None):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


OK_NOUL = {
    "model": "jev-1.13.0",
    "answers": {"answer": {"type": "noul", "noul": 0.95}},
    "usage": {"input_tokens": 296, "output_tokens": 20},
}
OK_CHOICE = {
    "model": "jev-1.13.0",
    "answers": {
        "answer": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
            "confidence": 0.81,
        }
    },
    "usage": {"input_tokens": 318, "output_tokens": 34},
}
OK_SCORE = {
    "model": "jev-1.13.0",
    "answers": {
        "answer": {
            "type": "score",
            "score": 1.05,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
            "confidence": 0.92,
        }
    },
    "usage": {"input_tokens": 304, "output_tokens": 18},
}


OK_OPENROUTER = {
    "id": "gen-abc123",
    "model": "typesafe/jev-1.13",
    "provider": "TypeSafe",
    "answers": {"answer": {"type": "noul", "noul": 0.97}},
    "usage": {"input_tokens": 312, "output_tokens": 20, "cost": 0.0000131},
}


async def test_noul():
    print("noul")
    t = tool()
    sent = stub(t, [OK_NOUL])
    out = await t.jev_yes_no(question="Is this urgent?", state="Payouts failing!")
    check("payload shape", sent[0]["questions"]["answer"]["type"] == "noul", sent[0])
    check("model passed", sent[0]["model"] == "jev-latest")
    check("state passed", sent[0]["state"] == "Payouts failing!")
    check("verdict yes", "**yes**" in out, out)
    check("probability shown", "0.95" in out, out)
    check("footer", "jev-1.13.0" in out, out)


async def test_choice():
    print("choice")
    t = tool()
    sent = stub(t, [OK_CHOICE])
    out = await t.jev_classify(
        question="Which team?",
        options={"billing": "Payments", "technical": "Bugs", "sales": "Pricing"},
        state="Payouts failing",
    )
    check("criteria forwarded", sent[0]["questions"]["answer"]["criteria"]["billing"] == "Payments")
    check("choice shown", "**billing**" in out, out)
    check("confidence shown", "confidence 0.81" in out, out)
    check("distribution sorted", out.index("billing 88%") < out.index("technical 12%"), out)
    check("zero prob dropped", "sales" not in out.split("\n")[1], out)


async def test_score():
    print("score")
    t = tool()
    sent = stub(t, [OK_SCORE])
    out = await t.jev_score(
        question="How frustrated?",
        levels=["Calm", "Frustrated", "Very angry"],
        state="Payouts failing",
    )
    check("criteria is a list", sent[0]["questions"]["answer"]["criteria"] == ["Calm", "Frustrated", "Very angry"])
    check("score shown", "1.05/2" in out, out)
    check("closest level", 'closest level: "Frustrated"' in out, out)
    check("legend in distribution", "1 Frustrated 95%" in out, out)


async def test_multi_question():
    print("ask_jev multi-question")
    t = tool()
    response = {
        "model": "jev-1.13.0",
        "answers": {
            "urgent": {"type": "noul", "noul": 0.9},
            "team": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "tech": 0.1},
                "confidence": 0.7,
            },
        },
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    sent = stub(t, [response])
    out = await t.ask_jev(
        questions={
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {
                "type": "choice",
                "instructions": "Who handles it?",
                "criteria": {"billing": "Payments", "tech": "Bugs"},
            },
        },
        state="Payouts failing",
    )
    check("both questions sent", set(sent[0]["questions"]) == {"urgent", "team"})
    check("single request", len(sent) == 1)
    check("labels used", "Is this urgent?" in out and "Who handles it?" in out, out)


async def test_json_string_args():
    print("stringified arguments (small models pass JSON strings)")
    t = tool()
    sent = stub(t, [OK_CHOICE])
    await t.jev_classify(
        question="Which team?",
        options='{"billing": "Payments", "technical": "Bugs", "sales": "Pricing"}',
        state="x",
    )
    check("options string parsed", len(sent[0]["questions"]["answer"]["criteria"]) == 3)

    t2 = tool()
    sent2 = stub(t2, [OK_SCORE])
    await t2.jev_score(question="How bad?", levels='["Low", "High"]', state="x")
    check("levels string parsed", sent2[0]["questions"]["answer"]["criteria"] == ["Low", "High"])

    t3 = tool()
    sent3 = stub(t3, [OK_NOUL])
    await t3.ask_jev(
        questions='{"a": {"type": "noul", "instructions": "Urgent?"}}', state="x"
    )
    check("questions string parsed", "a" in sent3[0]["questions"])

    t4 = tool()
    sent4 = stub(t4, [OK_CHOICE])
    await t4.jev_classify(question="Which?", options=["billing", "technical"], state="x")
    check("option list accepted", sent4[0]["questions"]["answer"]["criteria"] == {"billing": "billing", "technical": "technical"})


async def test_validation():
    print("validation")
    t = tool()
    stub(t, [OK_CHOICE])
    out = await t.jev_classify(question="Which?", options={"only": "one"}, state="x")
    check("rejects 1 option", "between 2 and 255" in out, out)

    out = await t.jev_score(question="How bad?", levels=[str(i) for i in range(11)], state="x")
    check("rejects 11 levels", "between 2 and 10" in out, out)

    out = await t.ask_jev(questions={"a": {"type": "vibes", "instructions": "hm"}}, state="x")
    check("rejects bad type", "expected one of" in out, out)

    out = await t.ask_jev(questions={"a": {"type": "noul"}}, state="x")
    check("requires instructions", "missing 'instructions'" in out, out)

    out = await t.ask_jev(questions={"a": {"type": "score", "instructions": "x"}}, state="x")
    check("score needs levels", "ordered list" in out, out)


async def test_state_resolution():
    print("state resolution")
    t = tool()
    sent = stub(t, [OK_NOUL])
    await t.jev_yes_no(
        question="Urgent?",
        __files__=[{"file": {"filename": "ticket.txt", "data": {"content": "payout broken"}}}],
    )
    check("file content used", "payout broken" in sent[0]["state"], sent[0]["state"])
    check("file name included", "ticket.txt" in sent[0]["state"])

    t2 = tool()
    sent2 = stub(t2, [OK_NOUL])
    await t2.jev_yes_no(
        question="Urgent?",
        __messages__=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
        ],
    )
    check("messages used", "user: hello" in sent2[0]["state"], sent2[0]["state"])
    check("multipart content flattened", "hi there" in sent2[0]["state"])

    t3 = tool(CONTEXT_MESSAGES=1)
    sent3 = stub(t3, [OK_NOUL])
    await t3.jev_yes_no(
        question="Urgent?",
        __messages__=[{"role": "user", "content": "old"}, {"role": "user", "content": "new"}],
    )
    check("context window honoured", "old" not in sent3[0]["state"], sent3[0]["state"])

    t4 = tool()
    stub(t4, [OK_NOUL])
    out = await t4.jev_yes_no(question="Urgent?")
    check("no state is an error", "nothing to evaluate" in out, out)

    t5 = tool()
    sent5 = stub(t5, [OK_NOUL])
    await t5.jev_yes_no(question="Urgent?", state="explicit", __files__=[{"file": {"data": {"content": "file text"}}}])
    check("explicit state wins over files", sent5[0]["state"] == "explicit")


async def test_truncation():
    print("truncation")
    t = tool(MAX_STATE_CHARS=200)
    sent = stub(t, [OK_NOUL])
    await t.jev_yes_no(question="Urgent?", state="x" * 5000)
    check("head kept, marker appended", sent[0]["state"].startswith("x") and sent[0]["state"].endswith("[...truncated...]"))
    check("length capped", len(sent[0]["state"]) < 260, len(sent[0]["state"]))

    t2 = tool(MAX_STATE_CHARS=200)
    sent2 = stub(t2, [OK_NOUL])
    await t2.jev_yes_no(
        question="Urgent?",
        __messages__=[{"role": "user", "content": "START" + "y" * 5000 + "END"}],
    )
    check("conversation keeps the tail", sent2[0]["state"].endswith("END"), sent2[0]["state"][-20:])


async def test_missing_key():
    print("missing API key")
    t = Tools()
    out = await t.jev_yes_no(question="Urgent?", state="x")
    check("clear message", "no API key configured" in out, out)


async def test_raw_output():
    print("raw JSON valve")
    t = tool(RAW_JSON_OUTPUT=True)
    stub(t, [OK_NOUL])
    out = await t.jev_yes_no(question="Urgent?", state="x")
    check("valid JSON returned", json.loads(out)["answers"]["answer"]["noul"] == 0.95)


async def test_batch():
    print("batch")
    t = tool(BATCH_CONCURRENCY=4)
    calls = []

    async def fake_send(session, payload, emitter):
        calls.append(payload["state"])
        if payload["state"] == "bad":
            raise JevError("HTTP 422 - nope")
        return OK_CHOICE

    t._send = fake_send
    out = await t.jev_classify_batch(
        question="Which team?",
        options={"billing": "Payments", "technical": "Bugs"},
        items=["one", "two", "bad", "four"],
    )
    check("one call per item", len(calls) == 4, calls)
    check("table rendered", out.count("|") > 10)
    check("failure row", "| 3 | bad | error | - |" in out, out)
    check("tally counts successes", "billing: 3" in out, out)
    check("failure reported", "Failed: 1" in out, out)
    check("token totals summed", "tokens in/out: 954/102" in out, out)

    t2 = tool(BATCH_MAX_ITEMS=2)
    out2 = await t2.jev_classify_batch(
        question="Which?", options={"a": "A", "b": "B"}, items=["1", "2", "3"]
    )
    check("batch cap enforced", "exceeds the batch limit" in out2, out2)

    t3 = tool()
    sent3 = stub(t3, [OK_CHOICE])
    await t3.jev_classify_batch(
        question="Which?", options={"a": "A", "b": "B"}, items="line one\n\nline two"
    )
    check("newline string split into items", len(sent3) == 2, sent3)


async def test_retry_loop():
    print("retry / error mapping (real _send)")
    t = tool(MAX_RETRIES=2, BACKOFF_FACTOR=0.0)

    session = FakeSession([
        FakeResponse(429, "rate limited", {"Retry-After": "0"}),
        FakeResponse(529, "overloaded"),
        FakeResponse(200, json.dumps(OK_NOUL)),
    ])
    data = await t._send(session, {"state": "x"}, None)
    check("retries 429 then 529 then succeeds", data["model"] == "jev-1.13.0")
    check("three attempts made", session.calls == 3, session.calls)

    session2 = FakeSession([FakeResponse(401, "bad key")])
    try:
        await t._send(session2, {"state": "x"}, None)
        check("401 raises", False)
    except JevError as exc:
        check("401 not retried", session2.calls == 1)
        check("401 message mentions valves", "JEV_API_KEY" in str(exc), str(exc))

    session3 = FakeSession([FakeResponse(422, "bad body")])
    try:
        await t._send(session3, {"state": "x"}, None)
        check("422 raises", False)
    except JevError as exc:
        check("422 not retried", session3.calls == 1)
        check("422 echoes body", "bad body" in str(exc), str(exc))

    session4 = FakeSession([FakeResponse(500, "boom")] * 3)
    try:
        await t._send(session4, {"state": "x"}, None)
        check("exhausted retries raise", False)
    except JevError as exc:
        check("all attempts used", session4.calls == 3, session4.calls)
        check("gives up with detail", "after 3 attempt" in str(exc), str(exc))

    session5 = FakeSession([FakeResponse(200, "<html>nope</html>")])
    try:
        await t._send(session5, {"state": "x"}, None)
        check("non-JSON 200 raises", False)
    except JevError as exc:
        check("non-JSON reported", "non-JSON" in str(exc), str(exc))

    t2 = tool(MAX_RETRIES=0)
    session6 = FakeSession([FakeResponse(429, "slow down")])
    try:
        await t2._send(session6, {"state": "x"}, None)
        check("zero retries raises", False)
    except JevError:
        check("MAX_RETRIES=0 means one attempt", session6.calls == 1, session6.calls)


class Recorder(FakeSession):
    """FakeSession that records the URL and headers of the last request."""

    def __init__(self, responses):
        super().__init__(responses)
        self.url = None
        self.headers = None

    def post(self, url, headers=None, json=None):
        self.url = url
        self.headers = headers
        return super().post(url, headers=headers, json=json)


async def test_url_and_headers():
    print("endpoint and auth header")
    t = tool(JEV_BASE_URL="https://proxy.internal/jev/")
    rec = Recorder([FakeResponse(200, json.dumps(OK_NOUL))])
    await t._send(rec, {"state": "x"}, None)
    check("trailing slash handled", rec.url == "https://proxy.internal/jev/v1/systemone", rec.url)
    check("bearer header", rec.headers["Authorization"] == "Bearer test-key")


async def test_openrouter_endpoint():
    print("OpenRouter Decisions endpoint")
    t = tool(
        JEV_BASE_URL="https://openrouter.ai",
        JEV_ENDPOINT_PATH="/api/alpha/decisions",
        JEV_MODEL="typesafe/jev-1.13",
    )
    rec = Recorder([FakeResponse(200, json.dumps(OK_OPENROUTER))])
    await t._send(rec, t._payload("x", {}), None)
    check("decisions URL built", rec.url == "https://openrouter.ai/api/alpha/decisions", rec.url)

    # the path valve should tolerate a missing or doubled leading slash
    for configured in ("api/alpha/decisions", "/api/alpha/decisions"):
        t2 = tool(JEV_BASE_URL="https://openrouter.ai/", JEV_ENDPOINT_PATH=configured)
        rec2 = Recorder([FakeResponse(200, json.dumps(OK_NOUL))])
        await t2._send(rec2, {"state": "x"}, None)
        check(f"path {configured!r} normalised", rec2.url == "https://openrouter.ai/api/alpha/decisions", rec2.url)

    t3 = tool(JEV_ENDPOINT_PATH="")
    rec3 = Recorder([FakeResponse(200, json.dumps(OK_NOUL))])
    await t3._send(rec3, {"state": "x"}, None)
    check("empty path falls back to native", rec3.url == "https://api.typesafe.ai/v1/systemone", rec3.url)

    t4 = tool(
        JEV_BASE_URL="https://openrouter.ai",
        JEV_ENDPOINT_PATH="/api/alpha/decisions",
        JEV_MODEL="typesafe/jev-1.13",
    )
    sent = stub(t4, [OK_OPENROUTER])
    out = await t4.jev_yes_no(question="Is this a bug?", state="blank screen after Pay")
    check("openrouter model name sent", sent[0]["model"] == "typesafe/jev-1.13", sent[0])
    check("openrouter answer rendered", "**yes**" in out, out)
    check("cost surfaced when returned", "cost: $0.000013" in out, out)


async def test_event_emitter():
    print("status events")
    t = tool()
    stub(t, [OK_NOUL])
    events = []

    async def emitter(event):
        events.append(event)

    await t.jev_yes_no(question="Urgent?", state="x", __event_emitter__=emitter)
    check("emits events", len(events) >= 2, events)
    check("final event done", events[-1]["data"]["done"] is True, events[-1])

    def broken(event):
        raise RuntimeError("emitter exploded")

    t2 = tool()
    stub(t2, [OK_NOUL])
    out = await t2.jev_yes_no(question="Urgent?", state="x", __event_emitter__=broken)
    check("broken emitter does not break the tool", "**yes**" in out, out)


async def main():
    for test in (
        test_noul,
        test_choice,
        test_score,
        test_multi_question,
        test_json_string_args,
        test_validation,
        test_state_resolution,
        test_truncation,
        test_missing_key,
        test_raw_output,
        test_batch,
        test_retry_loop,
        test_url_and_headers,
        test_openrouter_endpoint,
        test_event_emitter,
    ):
        await test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
