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
    monkeypatch.setattr(server, "checkpoint_profiles", lambda: {})

    async def fake_client_for(name: str):
        return ScriptedClient(profile)

    monkeypatch.setattr(server, "client_for", fake_client_for)
    monkeypatch.setenv("CODEQA_WARM_CLIENTS", "0")
    c = TestClient(server.app)
    c.headers["Authorization"] = "Bearer Action!"
    return c


def test_repos_lists_indexed_repos_and_hides_nodoc(client: TestClient) -> None:
    rows = client.get("/repos").json()
    assert [r["repo_id"] for r in rows] == [REPO_ID]
    assert rows[0] == {"repo_id": REPO_ID, "url": "https://github.com/acme/widgets", "sha": "0123abc0123abc0123abc0123abc0123abc01234",
                       "files": 1, "lines": 10, "symbols": 0, "stage": "ready", "summaries": False}


def test_profiles(client: TestClient) -> None:
    assert client.get("/profiles").json() == [{"name": "scripted", "kind": "anthropic", "model": "scripted", "label": "scripted", "note": "judge", "source": "profiles.yaml", "default": False}]


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
    assert events[5]["format_ok"] is True and events[5]["format_reason"] == ""
    assert all("t" in e for e in events[:6]) and events[4]["model_seconds"] >= 0 and events[4]["tool_seconds"] >= 0
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
    assert r["repo_id"] == REPO_ID and len(r["questions"]) == 5 and [i["type"] for i in r["items"]] == ["locate", "value", "enumerate", "trace", "explain"]
    assert "widgets" in r["questions"][0]
    assert client.get("/repos/x__y__0000000/suggestions").status_code == 404


def test_password_gate(client: TestClient) -> None:
    bare = TestClient(client.app)
    assert bare.get("/repos").status_code == 401
    assert bare.get("/health").status_code == 200
    assert bare.post("/auth/login", json={"password": "nope"}).status_code == 401
    assert bare.post("/auth/login", json={"password": "Action!"}).json() == {"ok": True}
    assert bare.get("/repos", headers={"X-Password": "Action!"}).status_code == 200


def test_checkpoint_profiles_are_listed_and_askable(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import apps.api.server as server
    logs = tmp_path / "logs"
    (logs / "r9").mkdir(parents=True)
    (logs / "r9" / "config.json").write_text(json.dumps({"model_name": "Qwen/Qwen3.5-4B", "renderer_name": "qwen3_5", "max_tokens": 1024}))
    (logs / "r9" / "checkpoints.jsonl").write_text(json.dumps({"name": "000002", "batch": 2, "sampler_path": "tinker://x/sampler_weights/000002"}) + "\n")
    monkeypatch.setattr(paths, "LOGS", logs)
    # override the fixture's stub with one synthesized checkpoint profile
    monkeypatch.setattr(server, "checkpoint_profiles", lambda: {"qwen4b-r9-step2": server.EndpointProfile(name="qwen4b-r9-step2", kind="tinker", model="tinker://x/sampler_weights/000002", base_model="Qwen/Qwen3.5-4B", renderer="qwen3_5")})
    rows = client.get("/profiles").json()
    ck = next(r for r in rows if r["name"] == "qwen4b-r9-step2")
    assert ck["source"] == "checkpoints" and ck["label"] == "r9 step 2" and "checkpoint" in ck["note"]
    assert client.post("/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "qwen4b-r9-step2"}).status_code == 200


def test_format_failure_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.api.server as server

    class NoCite(ScriptedClient):
        async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0):
            return Message(role="assistant", content="Line two says so, trust me.", usage={"prompt_tokens": 10, "completion_tokens": 5})

    async def fake(name: str):
        return NoCite(server.EndpointProfile(name="scripted", kind="anthropic", model="scripted"))

    monkeypatch.setattr(server, "client_for", fake)
    with client.stream("POST", "/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "scripted"}) as r:
        events = read_sse(r.read().decode())
    cit = next(e for e in events if e["type"] == "citations")
    assert cit["format_ok"] is False and "citation" in cit["format_reason"].lower()


def test_dead_tinker_session_is_reopened_once(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tinker answers 400 "cancelled" for every call once it has ended our session. The ask must drop the session,
    open a new one and answer on it, with no error event reaching the UI."""
    import apps.api.server as server

    class DeadThenAlive(ScriptedClient):
        dead = True

        async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0):
            if DeadThenAlive.dead:
                raise RuntimeError("Error code: 400 - {'detail': 'This ServiceClient has finished (interrupted) and cannot run further operations.'}")
            return await super().chat(messages, tools, max_tokens, temperature)

    profile = server.EndpointProfile(name="scripted", kind="tinker", model="Qwen/Qwen3.5-4B")
    resets: list[str] = []

    def fake_reset() -> None:
        resets.append("reset")
        DeadThenAlive.dead = False

    async def fake(name: str):
        return DeadThenAlive(profile)

    monkeypatch.setattr(server, "client_for", fake)
    monkeypatch.setattr(server, "reset_tinker_session", fake_reset)
    with client.stream("POST", "/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "scripted"}) as r:
        events = read_sse(r.read().decode())
    assert resets == ["reset"]
    assert [e["type"] for e in events if e["type"] == "error"] == []
    assert any(e["type"] == "answer" for e in events)
    assert next(e for e in events if e["type"] == "stats")["stop_reason"] == "answer"


def test_reset_tinker_session_drops_only_tinker_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.api.server as server

    class C:
        def __init__(self, kind: str) -> None:
            self.profile = server.EndpointProfile(name=kind, kind=kind, model="m")

    monkeypatch.setattr(server, "_clients", {"claude": C("anthropic"), "qwen4b-base": C("tinker")})
    server.reset_tinker_session()
    assert list(server._clients) == ["claude"]


def test_idle_tinker_client_is_probed_and_rebuilt_on_hang(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampling client whose probe hangs is rebuilt before the episode runs; one that answers is kept."""
    import apps.api.server as server

    profile = server.EndpointProfile(name="scripted", kind="tinker", model="Qwen/Qwen3.5-4B")
    built: list[str] = []
    probes: list[bool] = [False, True]   # first client hangs, the rebuilt one answers

    async def fake_client_for(name: str):
        built.append(name)
        return ScriptedClient(profile)

    async def fake_probe(c):
        return probes.pop(0)

    rebuilt: list[str] = []
    monkeypatch.setattr(server, "load_profiles", lambda: {"scripted": profile})
    monkeypatch.setattr(server, "client_for", fake_client_for)
    monkeypatch.setattr(server, "probe_tinker", fake_probe)
    monkeypatch.setattr(server, "rebuild_sampling_client", lambda name: rebuilt.append(name))
    monkeypatch.setattr(server, "_tinker_last_ok", {})
    with client.stream("POST", "/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "scripted"}) as r:
        events = read_sse(r.read().decode())
    assert rebuilt == ["scripted"] and built == ["scripted", "scripted"]
    assert any(e["type"] == "answer" for e in events)
    # the episode answered, so the profile is marked fresh and the next ask skips the probe
    assert "scripted" in server._tinker_last_ok
    with client.stream("POST", "/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "scripted"}) as r:
        read_sse(r.read().decode())
    assert rebuilt == ["scripted"] and probes == [True]


def test_v3_profiles_get_the_round_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bash_v3 checkpoint is served with the v3 caps (no call cap, context + message caps, several commands per
    message) and the bash knobs; other variants keep the task type's default budget."""
    import apps.api.server as server
    from codeqa.agent import shell

    v3 = server.EndpointProfile(name="qwen4b-p6_bash_v3-step20", kind="tinker", model="tinker://x", base_model="Qwen/Qwen3.5-4B", variant="bash_v3")
    lean = server.EndpointProfile(name="qwen4b-p6_full-step12", kind="tinker", model="tinker://y", base_model="Qwen/Qwen3.5-4B", variant="full")
    b = server.harness_budget(v3, "explain")
    assert b is not None and b.rounds_mode and b.max_tool_calls == server.UNLIMITED_CALLS
    assert (b.max_context_tokens, b.max_turns, b.max_commands_per_turn) == (server.V3_CONTEXT_TOKENS, server.V3_MESSAGES, server.V3_COMMANDS)
    assert b.max_answer_tokens == server.DEFAULT_BUDGETS["explain"].max_answer_tokens
    assert server.harness_budget(lean, "explain") is None
    server.apply_harness_knobs(v3)
    assert shell.heal_enabled() and shell.pipelines_enabled()
    server.apply_harness_knobs(lean)
    assert not shell.heal_enabled() and not shell.pipelines_enabled()
    row = server.harness_row(v3)
    assert row["variant"] == "bash_v3" and row["tools"] == ["bash"] and row["harness"]["rounds"] is True


def test_checkpoint_profiles_carry_the_run_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import apps.api.server as server
    from apps.api import workshop

    logs = tmp_path / "logs"
    (logs / "p9_bash_v3").mkdir(parents=True)
    (logs / "p9_bash_v3" / "checkpoints.jsonl").write_text(json.dumps({"name": "000004", "batch": 4, "sampler_path": "tinker://a/sampler_weights/000004"}) + "\n")
    (logs / "p9_bash_v3" / "config.json").write_text(json.dumps({"model_name": "Qwen/Qwen3.5-4B", "variant": None}))
    (logs / "runs.json").write_text(json.dumps({"p9_bash_v3": {"title": "v3", "variant": "bash_v3"}}))
    monkeypatch.setattr(server.paths, "LOGS", logs)
    monkeypatch.setattr(workshop.paths, "LOGS", logs)
    monkeypatch.setattr(server, "load_profiles", lambda: {})
    profs = server.checkpoint_profiles()
    assert profs["qwen4b-p9_bash_v3-step4"].variant == "bash_v3"


def test_tool_console_takes_a_variant(client: TestClient) -> None:
    r = client.get(f"/repos/{REPO_ID}/tools", params={"variant": "bash_v3"}).json()
    assert r["variant"] == "bash_v3" and [t["name"] for t in r["tools"]] == ["bash"] and "default" in r["variants"]
    assert client.get(f"/repos/{REPO_ID}/tools", params={"variant": "nope"}).status_code == 400
    r = client.post(f"/repos/{REPO_ID}/tool", json={"name": "bash", "args": {"command": "ls"}, "variant": "bash_v3"})
    assert r.status_code == 200 and "pkg" in r.json()["output"]
    r = client.post(f"/repos/{REPO_ID}/tool", json={"name": "bash", "args": {"command": "ls"}})
    assert r.status_code == 400   # bash is not in the default variant


def test_ask_variant_override_puts_any_profile_on_the_v3_harness(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`variant` on /ask runs the episode under that agent's tools, prompt and budget regardless of the profile."""
    import apps.api.server as server
    seen: dict[str, object] = {}

    class Peek(ScriptedClient):
        async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0):
            seen["tools"] = [t["name"] for t in (tools or [])]
            seen["system"] = messages[0].content
            return Message(role="assistant", content="Answer [pkg/mod.py:L2-L3].", usage={"prompt_tokens": 10, "completion_tokens": 5})

    async def fake(name: str):
        return Peek(server.EndpointProfile(name="scripted", kind="anthropic", model="scripted"))

    monkeypatch.setattr(server, "client_for", fake)
    with client.stream("POST", "/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "scripted", "variant": "bash_v3"}) as r:
        events = read_sse(r.read().decode())
    assert seen["tools"] == ["bash"]
    assert "one tool: bash" in str(seen["system"]) and "32k" in str(seen["system"])
    assert any(e["type"] == "answer" for e in events)
    r = client.post("/ask", json={"repo_id": REPO_ID, "question": "q", "profile": "scripted", "variant": "nope"})
    assert r.status_code == 400
