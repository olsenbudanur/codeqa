"""v3 harness (2026-09-20): rounds helpers, grep self-healing, the training env subclass, and the driver in rounds mode.
Offline, flask snapshot."""
from __future__ import annotations

import asyncio
import json

import pytest
from tinker_cookbook.renderers.base import ToolCall as CbToolCall

from codeqa.agent import rounds, shell
from codeqa.agent.driver import run_episode
from codeqa.agent.env import RepoEnv
from codeqa.agent.tool_env import CodeQAToolMessageEnv
from codeqa.agent.tools import RepoTools
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Message, Task, ToolCall

REPO_ID = "pallets__flask__85c5d93"
pytestmark = pytest.mark.skipif(not (paths.repo_dir(REPO_ID) / "manifest.json").exists(), reason="flask snapshot missing")
PROFILE = EndpointProfile(name="p", kind="tinker", model="Qwen/Qwen3.5-4B", variant="bash_v3")


def task() -> Task:
    return Task(task_id="t", repo_id=REPO_ID, question="Where is Flask defined?", task_type="locate", source="structural")


@pytest.fixture
def v3(monkeypatch):
    monkeypatch.setenv("CODEQA_CAPS_CONTEXT", "32000")
    monkeypatch.setenv("CODEQA_CAPS_MESSAGES", "4")
    monkeypatch.setenv("CODEQA_CAPS_COMMANDS", "2")
    monkeypatch.setenv("CODEQA_BASH_HEAL", "1")
    monkeypatch.setenv("CODEQA_SEEN_PIPELINES", "1")
    monkeypatch.setenv("CODEQA_BASH_EXECUTOR", "local")
    monkeypatch.setattr(shell, "EXECUTOR", "local")


# ---------------------------------------------------------------- pure helpers
def test_cap_calls_and_outputs():
    assert rounds.cap_calls([1, 2, 3], 2) == ([1, 2], 1)
    assert rounds.cap_calls([1], None) == ([1], 0)
    out = rounds.cap_outputs(["a" * 100, "b" * 100, "c" * 100], 150)
    assert out[0] == "a" * 100 and out[1].startswith("b" * 50) and "cut at 150" in out[1] and "cut at 150" in out[2]


def test_should_force_on_context_or_last_message():
    assert not rounds.should_force(1000, 32000, 1, 24)
    assert rounds.should_force(32000, 32000, 1, 24)
    assert rounds.should_force(100, 32000, 23, 24)


# ---------------------------------------------------------------- grep self-healing
def bash(t: RepoTools, command: str) -> str:
    from tinker_cookbook.tool_use.types import ToolInput
    return asyncio.run(t.bash.run(ToolInput(arguments={"command": command}))).messages[0]["content"]


def test_bad_regex_heals_to_literal(v3):
    out = bash(RepoTools(REPO_ID), "grep -rn 'class Flask\\(' src/flask")
    assert out.startswith("(the pattern was not a valid regex; showing literal matches")
    assert "src/flask/app.py:" in out and "class Flask(" in out


def test_zero_hits_heals_case_insensitive(v3):
    out = bash(RepoTools(REPO_ID), "grep -n 'CLASS FLASK(' src/flask/app.py")
    assert out.startswith("(no exact matches; showing case-insensitive matches)")
    assert "class Flask(" in out


def test_heal_off_by_default(monkeypatch):
    monkeypatch.delenv("CODEQA_BASH_HEAL", raising=False)
    monkeypatch.setattr(shell, "EXECUTOR", "local")
    out = bash(RepoTools(REPO_ID), "grep -n 'CLASS FLASK(' src/flask/app.py")
    assert "case-insensitive" not in out


def test_healed_hits_are_grounded(v3):
    t = RepoTools(REPO_ID)
    bash(t, "grep -rn 'class Flask\\(' src/flask")
    assert any(s.path == "src/flask/app.py" for s in t.files_read)


# ---------------------------------------------------------------- training env subclass
def cb_call(command: str, i: int = 0) -> CbToolCall:
    return CbToolCall(id=f"c{i}", function=CbToolCall.FunctionBody(name="bash", arguments=json.dumps({"command": command})))


def make_env(max_turns: int = 4, context_cap: int = 32000, commands: int = 2) -> tuple[CodeQAToolMessageEnv, RepoTools]:
    tools = RepoTools(REPO_ID, max_tool_calls=10**6)

    async def reward_fn(history):
        return 0.0, {}

    env = CodeQAToolMessageEnv(tools=tools.tools(("bash",)), initial_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
                               max_turns=max_turns, reward_fn=reward_fn, max_tool_calls=10**6, renderer=None,
                               context_cap=context_cap, commands_per_turn=commands)
    asyncio.run(env.initial_observation())
    return env, tools


def test_env_round_caps_commands_and_adds_trailer():
    env, tools = make_env()
    msg = {"role": "assistant", "content": "", "tool_calls": [cb_call("ls src/flask", 0), cb_call("wc -l src/flask/app.py", 1), cb_call("ls", 2)]}
    res = asyncio.run(env.step(msg))
    assert not res.episode_done
    tool_msgs = [m for m in env.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 2 and tools.calls == 2                    # third command not run
    last = tool_msgs[-1]["content"]
    assert "were not run: at most 2 commands per message" in last
    assert "[context " in last and "message(s) left]" in last
    assert env.history[2]["tool_calls"] is msg["tool_calls"]           # the model's own message is kept intact in history


def test_env_forces_final_answer_and_accepts_it():
    env, _ = make_env(max_turns=3)
    asyncio.run(env.step({"role": "assistant", "content": "", "tool_calls": [cb_call("ls src/flask")]}))   # turn 1
    assert env.history[-1]["role"] == "tool"
    res = asyncio.run(env.step({"role": "assistant", "content": "", "tool_calls": [cb_call("ls src")]}))    # turn 2 = max_turns - 1
    assert not res.episode_done and env.history[-1]["role"] == "user" and env.history[-1]["content"].startswith(rounds.FORCED_MARK)
    assert res.metrics.get("forced_answer_prompted") == 1.0
    res = asyncio.run(env.step({"role": "assistant", "content": "Flask is defined in [src/flask/app.py:L1-L2]"}))   # forced turn answers
    assert res.episode_done and res.metrics.get("stop/completed") == 1.0


def test_env_forced_turn_with_tools_is_no_answer():
    env, _ = make_env(max_turns=3)
    asyncio.run(env.step({"role": "assistant", "content": "", "tool_calls": [cb_call("ls src/flask")]}))
    asyncio.run(env.step({"role": "assistant", "content": "", "tool_calls": [cb_call("ls src")]}))
    res = asyncio.run(env.step({"role": "assistant", "content": "", "tool_calls": [cb_call("ls")]}))
    assert res.episode_done and res.metrics.get("forced_no_answer") == 1.0 and res.metrics.get("max_turns") == 1.0


def test_env_forces_on_context_cap():
    env, _ = make_env(max_turns=10, context_cap=200)     # renderer=None -> chars/4 estimate; one ls output passes 200 tokens
    res = asyncio.run(env.step({"role": "assistant", "content": "", "tool_calls": [cb_call("grep -rn 'def ' src/flask/app.py | head -80")]}))
    assert not res.episode_done and env.history[-1]["content"].startswith(rounds.FORCED_MARK)


def test_trace_from_history_marks_forced_answer(v3):
    renv = RepoEnv(task(), PROFILE)
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
               {"role": "assistant", "content": "", "tool_calls": [cb_call("ls")]}, {"role": "tool", "content": "x", "tool_call_id": "c0"},
               {"role": "user", "content": rounds.FORCED_PROMPT},
               {"role": "assistant", "content": "Answer [src/flask/app.py:L1-L2]"}]
    tr = renv.trace_from_history(history)
    assert tr.stats.forced_answer and tr.stats.stop_reason == "answer" and tr.answer.startswith("Answer")


# ---------------------------------------------------------------- driver (product / eval path)
class ScriptedClient:
    def __init__(self, replies: list[Message]):
        self.profile = PROFILE
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0) -> Message:
        self.seen.append(list(messages))
        m = self.replies.pop(0)
        m.usage = {"prompt_tokens": 1000 * len(self.seen), "completion_tokens": 50}
        return m


def test_driver_rounds_mode_trailer_and_forced_answer(v3):
    env = RepoEnv(task(), PROFILE)
    assert env.budget.rounds_mode and env.budget.max_turns == 4
    client = ScriptedClient([
        Message(role="assistant", tool_calls=[ToolCall(name="bash", args={"command": "ls src/flask"}), ToolCall(name="bash", args={"command": "ls"}),
                                              ToolCall(name="bash", args={"command": "wc -l src/flask/app.py"})]),
        Message(role="assistant", tool_calls=[ToolCall(name="bash", args={"command": "ls src"})]),
        Message(role="assistant", tool_calls=[ToolCall(name="bash", args={"command": "ls src"})]),      # turn 3 = max_turns - 1 -> forced prompt
        Message(role="assistant", content="Flask lives in app.py [src/flask/app.py:L1-L2]"),
    ])
    trace = asyncio.run(run_episode(env, client))
    tools = [m for m in trace.messages if m.role == "tool"]
    assert len(tools) == 4                                             # 2 (capped from 3) + 1 + 1
    assert "not run: at most 2 commands per message" in tools[1].content and "message(s) left]" in tools[1].content
    assert any(m.role == "user" and m.content.startswith(rounds.FORCED_MARK) for m in trace.messages)
    assert trace.stats.stop_reason == "answer" and trace.stats.forced_answer and trace.stats.tool_calls == 4


def test_driver_rounds_mode_forced_turn_with_tools_is_no_answer(v3):
    env = RepoEnv(task(), PROFILE)
    client = ScriptedClient([Message(role="assistant", tool_calls=[ToolCall(name="bash", args={"command": "ls"})]) for _ in range(4)])
    trace = asyncio.run(run_episode(env, client))
    assert trace.stats.stop_reason == "max_turns" and not trace.answer and not trace.stats.forced_answer


# ---------------------------------------------------------------- unnumbered pipeline reads (CODEQA_SEEN_PIPELINES=1)
@pytest.mark.parametrize("cmd,start,end", [
    ("cat src/flask/app.py | head -150", 1, 150),
    ("cat src/flask/app.py | head -180 | tail -30", 151, 180),
    ("head -100 src/flask/app.py | tail -20", 81, 100),
    ("sed -n '40,60p' src/flask/app.py | head -5", 40, 44),
])
def test_pipeline_reads_register_their_range(v3, cmd, start, end):
    t = RepoTools(REPO_ID)
    bash(t, cmd)
    spans = [s for s in t.files_read if s.path == "src/flask/app.py"]
    assert spans and spans[0].start == start and spans[0].end == end, (cmd, spans)


def test_pipeline_reads_off_by_default(monkeypatch):
    monkeypatch.delenv("CODEQA_SEEN_PIPELINES", raising=False)
    monkeypatch.setattr(shell, "EXECUTOR", "local")
    t = RepoTools(REPO_ID)
    bash(t, "cat src/flask/app.py | head -150")
    assert not t.files_read


def test_pipeline_with_grep_stage_registers_nothing_unnumbered(v3):
    t = RepoTools(REPO_ID)
    bash(t, "cat src/flask/app.py | grep Flask")
    assert not t.files_read
