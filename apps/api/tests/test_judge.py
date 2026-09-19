"""The Opus referee flow with scripted clients: research through the harness, then one verdict per candidate."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from codeqa.shared.contracts import Message, ToolCall
from apps.api.tests.test_server import REPO_ID, ScriptedClient, data_dir, read_sse  # noqa: F401  (fixtures)


class RefereeResearch(ScriptedClient):
    """Reads lines 1–5, then answers with a verified citation."""


class RefereeVerdict:
    profile = None

    async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0) -> Message:
        user = messages[-1].content
        score = 9 if 'label "good"' in user else 2
        return Message(role="assistant", content=json.dumps({"score": score, "correct": score > 5, "summary": "judged", "issues": [] if score > 5 else ["no citations"], "strengths": ["cites L2"] if score > 5 else []}))


@pytest.fixture
def client(data_dir, monkeypatch: pytest.MonkeyPatch) -> TestClient:  # noqa: F811
    import apps.api.judge as judge
    import apps.api.server as server
    from codeqa.shared.contracts import EndpointProfile

    monkeypatch.setattr(judge, "research_client", lambda: RefereeResearch(EndpointProfile(name="opus", kind="anthropic", model="x")))
    monkeypatch.setattr(judge, "verdict_client", lambda: RefereeVerdict())
    monkeypatch.setenv("CODEQA_WARM_CLIENTS", "0")
    c = TestClient(server.app)
    c.headers["Authorization"] = "Bearer Action!"
    return c


def test_judge_streams_research_then_one_verdict_per_candidate(client: TestClient) -> None:
    body = {
        "repo_id": REPO_ID,
        "question": "What does line two say?",
        "candidates": [
            {"label": "good", "answer": "Line two says so [pkg/mod.py:L2-L3].", "citations": [{"path": "pkg/mod.py", "start": 2, "end": 3, "verified": True}]},
            {"label": "bad", "answer": "Trust me.", "citations": []},
        ],
    }
    with client.stream("POST", "/judge", json=body) as r:
        assert r.status_code == 200
        frames = read_sse(r.read().decode())
    types = [f["type"] for f in frames]
    assert types[0] == "phase" and frames[0]["phase"] == "research"
    assert [f["event"]["type"] for f in frames if f["type"] == "ref"] == ["thinking", "tool_call", "tool_result", "answer", "stats"]
    judging = next(f for f in frames if f["type"] == "phase" and f["phase"] == "judging")
    assert judging["reference"].startswith("Line two") and judging["ref_calls"] == 1
    verdicts = sorted((f for f in frames if f["type"] == "verdict"), key=lambda f: f["index"])
    assert [v["label"] for v in verdicts] == ["good", "bad"]
    assert verdicts[0]["score"] == 9 and verdicts[0]["correct"] is True
    assert verdicts[1]["score"] == 2 and verdicts[1]["issues"] == ["no citations"]
    assert types[-1] == "done"


def test_judge_requires_password_and_known_repo(client: TestClient) -> None:
    assert TestClient(client.app).post("/judge", json={"repo_id": REPO_ID, "question": "q", "candidates": [{"label": "a", "answer": "x"}]}).status_code == 401
    assert client.post("/judge", json={"repo_id": "x__y__0000000", "question": "q", "candidates": [{"label": "a", "answer": "x"}]}).status_code == 404


def test_judge_grades_each_candidate_with_the_training_grader(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate whose /ask trace is on disk gets a `grade` frame from the real grader (reference = the referee's
    answer, KeywordJudge stands in for the rubric judge); a candidate without a trace gets an error frame."""
    import apps.api.judge as judge
    from codeqa.grader.judge import KeywordJudge
    from codeqa.shared import paths
    from codeqa.shared.contracts import Trace, TraceStats, Span

    q = "What does line two say?"
    tid = judge.adhoc_task_id(REPO_ID, q)
    d = paths.TRACES / "product"
    d.mkdir(parents=True, exist_ok=True)
    good = Trace(task_id=tid, profile="good-model", messages=[Message(role="assistant", content="Line two says so [pkg/mod.py:L2-L3].", usage={"prompt_tokens": 100, "completion_tokens": 20})],
                 stats=TraceStats(tool_calls=1, files_read=[Span(path="pkg/mod.py", start=1, end=5)], stop_reason="answer", turns=2), answer="Line two says so [pkg/mod.py:L2-L3].")
    (d / f"{tid}__good-model.json").write_text(good.model_dump_json())
    monkeypatch.setattr(judge, "default_client", lambda model: KeywordJudge(), raising=False)
    import codeqa.grader.judge as gj
    monkeypatch.setattr(gj, "default_client", lambda model=None: KeywordJudge())

    body = {"repo_id": REPO_ID, "question": q, "variant": "bash_v3",
            "candidates": [{"label": "good", "profile": "good-model", "answer": good.answer, "citations": [{"path": "pkg/mod.py", "start": 2, "end": 3, "verified": True}]},
                           {"label": "bad", "answer": "Trust me.", "citations": []}]}
    with client.stream("POST", "/judge", json=body) as r:
        frames = read_sse(r.read().decode())
    grades = {f["index"]: f for f in frames if f["type"] == "grade"}
    assert set(grades) == {0, 1}
    assert grades[0]["reward"] is not None and grades[0]["components"]["citations_grounded"] == 1.0 and grades[0]["gate_failed"] is None
    assert "no profile" in grades[1]["error"]
    assert client.post("/judge", json={**body, "variant": "nope"}).status_code == 400
