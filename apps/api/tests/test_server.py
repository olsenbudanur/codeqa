"""apps/api on a throwaway repo and a scripted model client: no network, no keys."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Message, ToolCall

REPO_ID = "acme__widgets__0123abc"
SRC = "".join(f"line {i}\n" for i in range(1, 11))


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repos, index = tmp_path / "repos", tmp_path / "index"
    for name in ("REPOS", "INDEX", "TRACES"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    (repos / REPO_ID / "pkg").mkdir(parents=True)
    (repos / REPO_ID / "pkg" / "mod.py").write_text(SRC)
    (repos / REPO_ID / "manifest.json").write_text(json.dumps({
        "repo_id": REPO_ID, "url": "https://github.com/acme/widgets", "sha": "0123abc0123abc0123abc0123abc0123abc01234",
        "files": [{"path": "pkg/mod.py", "lang": "python", "lines": 10, "bytes": len(SRC)}], "dropped": {},
    }))
    (index / REPO_ID).mkdir(parents=True)
    (index / REPO_ID / "symbols.json").write_text(json.dumps({"repo_id": REPO_ID, "symbols": []}))
    (index / REPO_ID / "map.txt").write_text("pkg/\n  mod.py\n")
    # a training-only variant that must stay hidden
    (repos / f"{REPO_ID}__nodoc").mkdir()
    (repos / f"{REPO_ID}__nodoc" / "manifest.json").write_text("{}")
    from codeqa.grader import repo as grepo
    grepo.load_repo.cache_clear()
    return tmp_path


class ScriptedClient:
    """First turn reads lines 1–5, second turn answers with one grounded and one ungrounded citation."""

    def __init__(self, profile: EndpointProfile) -> None:
        self.profile = profile
        self.turn = 0

    async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0) -> Message:
        self.turn += 1
        if self.turn == 1:
            return Message(role="assistant", thinking="Start with the module.",
                           tool_calls=[ToolCall(name="read_file", args={"path": "pkg/mod.py", "start": 1, "end": 5})],
                           usage={"prompt_tokens": 100, "completion_tokens": 20})
        return Message(role="assistant", content="Line two says so [pkg/mod.py:L2-L3]. Line nine too [pkg/mod.py:L9-L9].",
                       usage={"prompt_tokens": 150, "completion_tokens": 30})


@pytest.fixture
def client(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import apps.api.server as server

    profile = EndpointProfile(name="scripted", kind="anthropic", model="scripted")
    monkeypatch.setattr(server, "load_profiles", lambda: {"scripted": profile})
    monkeypatch.setattr(server, "get_profile", lambda name: profile)

    async def fake_client_for(name: str):
        return ScriptedClient(profile)

    monkeypatch.setattr(server, "client_for", fake_client_for)
    monkeypatch.setenv("CODEQA_WARM_CLIENTS", "0")
    return TestClient(server.app)


def test_repos_lists_indexed_repos_and_hides_nodoc(client: TestClient) -> None:
    rows = client.get("/repos").json()
    assert [r["repo_id"] for r in rows] == [REPO_ID]
    assert rows[0] == {"repo_id": REPO_ID, "url": "https://github.com/acme/widgets", "sha": "0123abc0123abc0123abc0123abc0123abc01234",
                       "files": 1, "lines": 10, "symbols": 0, "stage": "ready", "summaries": False}


def test_profiles(client: TestClient) -> None:
    assert client.get("/profiles").json() == [{"name": "scripted", "kind": "anthropic", "model": "scripted", "label": "scripted", "note": "judge"}]


def test_file_reads_ranges_and_refuses_escapes(client: TestClient) -> None:
    assert client.get("/file", params={"repo_id": REPO_ID, "path": "pkg/mod.py"}).text == SRC
    assert client.get("/file", params={"repo_id": REPO_ID, "path": "pkg/mod.py", "start": 2, "end": 3}).text == "line 2\nline 3\n"
    assert client.get("/file", params={"repo_id": REPO_ID, "path": "../../etc/passwd"}).status_code == 404
    assert client.get("/file", params={"repo_id": "nope__nope__0000000", "path": "x"}).status_code == 404


def test_post_repo_rejects_non_github(client: TestClient) -> None:
    assert client.post("/repos", json={"url": "https://example.com/x"}).status_code == 400


def read_sse(text: str) -> list[dict]:
    return [json.loads(f[len("data: "):]) for f in text.strip().split("\n\n") if f.startswith("data: ")]


def test_ask_streams_c9_events_with_grader_citations(client: TestClient) -> None:
    with client.stream("POST", "/ask", json={"repo_id": REPO_ID, "question": "What does line two say?", "profile": "scripted"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = read_sse(r.read().decode())
    types = [e["type"] for e in events]
    assert types == ["thinking", "tool_call", "tool_result", "answer", "stats", "citations", "done"]
    assert events[1]["name"] == "read_file" and events[1]["args"]["start"] == 1
    assert events[3]["markdown"].startswith("Line two")
    cits = events[5]["items"]
    assert cits == [
        {"path": "pkg/mod.py", "start": 2, "end": 3, "exists": True, "verified": True},
        {"path": "pkg/mod.py", "start": 9, "end": 9, "exists": True, "verified": False},
    ]
    assert events[4]["tool_calls"] == 1 and events[4]["prompt_tokens"] == 250
    traces = list((paths.TRACES / "product").glob("*.json"))
    assert len(traces) == 1


def test_ask_unknown_repo_or_profile(client: TestClient) -> None:
    assert client.post("/ask", json={"repo_id": "x__y__0000000", "question": "q", "profile": "scripted"}).status_code == 404
    assert client.post("/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "nope"}).status_code == 404


def test_post_repo_starts_a_job(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.api.repos as repos

    async def fake_run(self, job):
        job.repo_id = "acme__other__abcdef0"
        job.stage = "ready"

    monkeypatch.setattr(repos.IndexJobs, "_run", fake_run)
    r = client.post("/repos", json={"url": "https://github.com/acme/other"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    s = client.get(f"/repos/{job_id}/status").json()
    assert s["stage"] == "ready" and s["repo_id"] == "acme__other__abcdef0"
    assert client.get("/repos/nope/status").status_code == 404


def test_suggestions_fall_back_to_generic_without_symbols(client: TestClient) -> None:
    r = client.get(f"/repos/{REPO_ID}/suggestions").json()
    assert r["repo_id"] == REPO_ID and len(r["questions"]) == 3
    assert "widgets" in r["questions"][0]
    assert client.get("/repos/x__y__0000000/suggestions").status_code == 404
