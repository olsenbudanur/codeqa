"""Workshop endpoints over a tiny synthetic data/ tree."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeqa.shared import paths

RUN = "unit_run"
REPO = "acme__widgets__0123abc"


def _w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("REPOS", "INDEX", "TRACES", "LOGS", "EVALS", "TASKS", "MODELS"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    monkeypatch.setattr(paths, "MODELS_MANIFEST", tmp_path / "models" / "manifest.json")
    import apps.api.workshop as ws
    ws._cache.clear()
    ws._repo_ids.cache_clear()
    # repo + index
    src = "".join(f"line {i}\n" for i in range(1, 21))
    _w(paths.REPOS / REPO / "pkg" / "mod.py", src)
    _w(paths.REPOS / REPO / "manifest.json", json.dumps({"repo_id": REPO, "url": "u", "sha": "0" * 40,
                                                          "files": [{"path": "pkg/mod.py", "lang": "python", "lines": 20, "bytes": 1}], "dropped": {}}))
    _w(paths.INDEX / REPO / "symbols.json", json.dumps({"repo_id": REPO, "symbols": []}))
    _w(paths.INDEX / REPO / "map.txt", "pkg/mod.py\n")
    # tasks + passrate
    task = {"task_id": "t-1", "repo_id": REPO, "split": "train", "question": "What is on line two?", "task_type": "locate", "source": "structural",
            "grading": {"expected_paths": ["pkg/mod.py"], "required_citations": [{"path": "pkg/mod.py", "start": 2, "end": 2}]}}
    _w(paths.TASKS / "train" / "all.jsonl", json.dumps(task) + "\n")
    _w(paths.TASKS / "reports" / "passrate.jsonl", json.dumps({"task_id": "t-1", "source": "structural", "task_type": "locate", "n": 4, "n_answered": 3,
                                                               "answered": 0.75, "format_ok": 0.5, "found": 1.0, "correct_lenient": 0.5, "reward": 0.25, "tool_calls": 3.0}) + "\n")
    # run: 3 steps, one iteration with one rollout
    _w(paths.LOGS / RUN / "config.json", json.dumps({"learning_rate": 1e-4, "model_name": "Qwen/Qwen3.5-4B",
                                                     "dataset_builder": {"group_size": 4, "groups_per_batch": 2, "tasks_path": "x", "profile_name": "qwen4b-base"}}))
    _w(paths.LOGS / RUN / "metrics.jsonl", "".join(json.dumps({"step": i, "env/all/reward/total": 0.1 * i, "env/all/group_reward_std": 0.2,
                                                                 "env/all/format_ok": 0.5, "time/total": 1.0, "time/policy_sample:mean": 2.0}) + "\n" for i in range(3)))
    rollout = {"iteration": 0, "group_idx": 0, "traj_idx": 1, "tags": ["structural", "locate"], "total_reward": 1.0, "sampling_client_step": 0,
               "trajectory_metrics": {"reward": 1.0, "format_ok": 1.0, "citations_grounded": 1.0, "correctness": 1.0, "efficiency": 1.0, "tool_calls": 1.0,
                                      "prompt_tokens": 100.0, "turns": 2.0, "stop_answer": 1.0},
               "steps": [{"logs": {"assistant_content": "\n", "tool_call_0": "read_file({\"path\": \"pkg/mod.py\", \"start\": 1, \"end\": 5})", "tool_result_0": "pkg/mod.py:L1-L5\nline 1"}},
                         {"logs": {"assistant_content": "Line two [pkg/mod.py:L2-L2]."}}]}
    _w(paths.LOGS / RUN / "iteration_000000" / "train_rollout_summaries.jsonl", json.dumps(rollout) + "\n")
    # dev trace (ungraded) + eval trace (graded)
    trace = {"task_id": "t-1", "profile": "claude", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "map\n\nQuestion: What is on line two?"},
        {"role": "assistant", "thinking": "Read it.", "tool_calls": [{"name": "read_file", "args": {"path": "pkg/mod.py", "start": 1, "end": 5}, "call_id": "c1"}]},
        {"role": "tool", "name": "read_file", "content": "pkg/mod.py:L1-L5\nline 1\nline 2", "call_id": "c1"},
        {"role": "assistant", "content": "Line two says so [pkg/mod.py:L2-L2] and nine [pkg/mod.py:L9-L9]."}],
        "stats": {"turns": 2, "tool_calls": 1, "prompt_tokens": 100, "completion_tokens": 10, "files_read": [{"path": "pkg/mod.py", "start": 1, "end": 5}],
                  "stop_reason": "answer", "seconds": 1.5},
        "answer": "Line two says so [pkg/mod.py:L2-L2] and nine [pkg/mod.py:L9-L9]."}
    _w(paths.TRACES / "dev" / "t-1__claude.json", json.dumps(trace))
    _w(paths.EVALS / "claude" / "fast" / "traces" / "t-1.json", json.dumps(trace))
    _w(paths.EVALS / "claude" / "fast" / "per_task.jsonl", json.dumps({"task_id": "t-1", "reward": 0.5, "gate_failed": None, "notes": "paths 1 gold: 1.00",
                                                                        "components": {"format_ok": 1.0, "correctness": 0.5}}) + "\n")
    _w(paths.EVALS / "claude" / "fast" / "results.json", json.dumps({"profile": "claude", "summary": {"n": 1, "reward": 0.5, "correct_rate": 0.5}}))
    _w(paths.MODELS_MANIFEST, json.dumps([{"name": "qwen4b-r-step1", "run": RUN, "step": 1, "created_at": "2026-09-18T00:00:00Z",
                                          "tinker_path": "tinker://x", "profile": "qwen4b-r-step1", "evals": {}}]))
    from codeqa.grader import repo as grepo
    grepo.load_repo.cache_clear()
    return tmp_path


@pytest.fixture
def client(data: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CODEQA_WARM_CLIENTS", "0")
    import apps.api.server as server
    c = TestClient(server.app)
    c.headers["Authorization"] = "Bearer Action!"
    return c


def test_runs(client: TestClient) -> None:
    runs = client.get("/runs").json()
    assert [r["name"] for r in runs] == [RUN] and runs[0]["steps"] == 3 and runs[0]["config"]["group_size"] == 4
    run = client.get(f"/runs/{RUN}").json()
    assert [m["env/all/reward/total"] for m in run["metrics"]] == [0.0, 0.1, 0.2]
    assert "time/total" in run["metrics"][0] and "time/policy_sample:mean" not in run["metrics"][0]
    it = client.get(f"/runs/{RUN}/iterations/0").json()
    t = it["groups"][0]["trajectories"][0]
    assert t["tool_sequence"] == ["read_file"] and t["reward"] == 1.0 and t["stop"] == "answer"
    assert client.get(f"/runs/{RUN}/iterations/9").status_code == 404
    assert client.get("/runs/nope").status_code == 404


def test_checkpoints_join_manifest_profiles_and_evals(client: TestClient) -> None:
    ck = client.get("/checkpoints").json()
    assert ck["checkpoints"][0]["name"] == "qwen4b-r-step1" and ck["checkpoints"][0]["servable"] is False
    claude = next(b for b in ck["baselines"] if b["name"] == "claude")
    assert claude["evals"]["fast"]["reward"] == 0.5 and ck["sets"] == ["fast"]


def test_data_pages(client: TestClient) -> None:
    s = client.get("/data/summary").json()
    assert s["counts"]["train/all"] == {"structural": {"locate": 1}} and s["repos"][0]["repo_id"] == REPO
    pr = client.get("/data/passrate?bins=10").json()
    assert pr["n"] == 1 and sum(pr["histogram"]) == 1 and pr["buckets"] == {"kept": 1}
    tk = client.get("/data/tasks?split=train").json()
    card = tk["tasks"][0]
    assert card["difficulty"] == 0.5 and card["bucket"] == "kept" and set(card["example_traces"]) == {"trace/dev/t-1__claude", "eval/claude/fast/t-1"}
    assert client.get("/data/tasks?split=train&bucket=too_hard").json()["total"] == 0


def test_traces_list_detail_compare(client: TestClient) -> None:
    lst = client.get("/traces").json()
    ids = {t["id"] for t in lst["traces"]}
    assert ids == {"trace/dev/t-1__claude", "eval/claude/fast/t-1", f"rollout/{RUN}/0/0/1"}
    assert client.get("/traces?has_citations=true").json()["total"] == 3
    assert client.get("/traces?run=eval:fast").json()["traces"][0]["reward"] == 0.5
    dev = client.get("/traces/trace/dev/t-1__claude").json()
    assert [e["type"] for e in dev["events"]] == ["thinking", "tool_call", "tool_result", "answer", "stats", "done"]
    assert dev["grade"]["source"] == "check_citations" and [c["verified"] for c in dev["citations"]] == [True, False]
    assert [m["role"] for m in dev["messages"]] == ["system", "user", "assistant", "tool", "assistant"] and dev["messages"][3]["content"].startswith("pkg/mod.py:L1-L5")
    ev = client.get("/traces/eval/claude/fast/t-1").json()
    assert ev["grade"]["reward"] == 0.5 and ev["grade"]["notes"] == "paths 1 gold: 1.00" and ev["question"] == "What is on line two?"
    ro = client.get(f"/traces/rollout/{RUN}/0/0/1").json()
    assert [e["type"] for e in ro["events"]] == ["tool_call", "tool_result", "answer", "stats", "done"] and ro["grade"]["reward"] == 1.0
    assert [m["role"] for m in ro["messages"]] == ["assistant", "tool", "assistant"] and ro["messages"][1]["name"] == "read_file" and "messages_note" in ro
    cmp = client.get("/traces/compare", params={"a": "trace/dev/t-1__claude", "b": "eval/claude/fast/t-1"}).json()
    assert cmp["a"]["profile"] == "claude" and cmp["b"]["grade"]["reward"] == 0.5
    assert client.get("/traces/trace/dev/nope").status_code == 404


def test_run_carries_optimizer_metrics_and_warnings(client: TestClient, data: Path) -> None:
    # A run whose entropy collapses and KL jumps must surface the monitor's warnings.
    rows = [{"step": i, "env/all/reward/total": 0.2, "env/all/reward": 0.2, "env/all/group_reward_std": 0.3, "optim/lr": 1e-4,
             "optim/entropy": 0.4 if i == 0 else 0.1, "optim/kl_sample_train_v1": 0.0 if i == 0 else 0.2, "kl_ref/kl": 0.01} for i in range(2)]
    (paths.LOGS / RUN / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    import apps.api.workshop as ws
    ws._cache.clear()
    run = client.get(f"/runs/{RUN}").json()
    assert run["metrics"][1]["optim/entropy"] == 0.1 and run["metrics"][1]["kl_ref/kl"] == 0.01
    assert any("KL" in w for w in run["warnings"]) and any("entropy" in w for w in run["warnings"])


def test_repo_overview_and_tool_console(client: TestClient) -> None:
    ov = client.get(f"/repos/{REPO}/overview").json()
    assert ov["n_files"] == 1 and ov["lines"] == 20 and ov["map"].startswith("pkg/") and ov["files"][0]["path"] == "pkg/mod.py"
    specs = client.get(f"/repos/{REPO}/tools").json()
    assert {t["name"] for t in specs["tools"]} == {"overview", "find_symbol", "grep", "read_file", "list_dir"}
    r = client.post(f"/repos/{REPO}/tool", json={"name": "read_file", "args": {"path": "pkg/mod.py", "start": 2, "end": 3}}).json()
    assert "line 2" in r["output"] and r["error"] is False and r["files_read"] == [{"path": "pkg/mod.py", "start": 2, "end": 3}]
    bad = client.post(f"/repos/{REPO}/tool", json={"name": "read_file", "args": {"path": "nope.py"}}).json()
    assert bad["error"] is True
    assert client.post(f"/repos/{REPO}/tool", json={"name": "rm", "args": {}}).status_code == 400
    assert client.get("/repos/x__y__0000000/overview").status_code == 404
