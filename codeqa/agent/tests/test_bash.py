"""bash variant: containment, seen-lines grounding, and the variant switch. Offline, flask snapshot."""
from __future__ import annotations

import asyncio

import pytest
from tinker_cookbook.tool_use.types import ToolInput

from codeqa.agent import shell
from codeqa.agent.env import RepoEnv
from codeqa.agent.tools import RepoTools
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Span, Task

REPO_ID = "pallets__flask__85c5d93"
pytestmark = pytest.mark.skipif(not (paths.repo_dir(REPO_ID) / "manifest.json").exists(), reason="flask snapshot missing")
PROFILE = EndpointProfile(name="p", kind="tinker", model="Qwen/Qwen3.5-4B")
TASK = Task(task_id="t", repo_id=REPO_ID, question="Where is Flask defined?", task_type="locate", source="structural")


def bash(t: RepoTools, command: str) -> str:
    return asyncio.run(t.bash.run(ToolInput(arguments={"command": command}))).messages[0]["content"]


@pytest.fixture
def t() -> RepoTools:
    return RepoTools(REPO_ID)


def test_grep_n_records_seen_lines(t):
    out = bash(t, "grep -n 'class Flask(' src/flask/app.py")
    assert "81:class Flask(App):" in out
    assert Span(path="src/flask/app.py", start=81, end=81) in t.files_read


def test_recursive_grep_records_paths(t):
    out = bash(t, "grep -rn 'def route' src/flask | head -5")
    assert ":L" not in out and "src/flask/" in out
    assert t.files_read and all(s.path.startswith("src/flask/") for s in t.files_read)


def test_nl_sed_records_a_range(t):
    out = bash(t, "nl -ba src/flask/app.py | sed -n '81,85p'")
    assert "81\tclass Flask(App):" in out
    assert Span(path="src/flask/app.py", start=81, end=85) in t.files_read


def test_unnumbered_single_file_read_records_its_range(t):
    """2026-09-20: `sed -n 'A,Bp' FILE` shows real lines; the model can count from A, so the range counts as seen."""
    bash(t, "sed -n '81,85p' src/flask/app.py")
    assert [(s.path, s.start, s.end) for s in t.files_read] == [("src/flask/app.py", 81, 85)]


def test_unnumbered_multi_file_output_records_nothing(t):
    bash(t, "grep -r 'import' src/flask/app.py src/flask/cli.py | head -5")     # no -n: which file/line is unknowable
    assert t.files_read == []


@pytest.mark.parametrize("cmd", ["cat ../../.env", "cat /etc/passwd", "cat ~/.zshrc", "sed -i 's/a/b/' src/flask/app.py",
                                 "find . -name '*.py' -delete", "echo x > src/flask/app.py", "rm -rf src", "python -c 'print(1)'",
                                 "cat $(echo src/flask/app.py)"])
def test_blocked_commands(t, cmd):
    out = bash(t, cmd)
    assert out.startswith("ERROR blocked"), out
    assert t.errors == 1 and t.files_read == []


def test_unknown_binary_is_not_available(t):
    out = bash(t, "awk '{print}' src/flask/app.py")
    assert "command not found" in out or "restricted" in out or out.startswith("(exit") or "only read-only commands" in out   # precheck rejects the command word


def test_output_cap(t):
    out = bash(t, "cat src/flask/app.py")
    assert "output truncated" in out and len(out) < shell.OUTPUT_CAP + 200


def test_no_match_is_not_an_error(t):
    out = bash(t, "grep -rn 'zzzqqq_nothing' src")
    assert out == "(no output)" and t.errors == 0


def test_precheck_allows_normal_pipelines():
    for cmd in ["grep -rn 'route' --include='*.py' . | head -20", "nl -ba src/flask/app.py | sed -n '1,40p'", "find src -name '*.py' | wc -l",
                "ls src/flask", "wc -l src/flask/app.py", "grep -n 'import' src/flask/app.py | cut -d: -f1 | head"]:
        assert shell.precheck(cmd) is None, cmd


def test_variants_change_tools_and_prompt_only(monkeypatch):
    monkeypatch.delenv("CODEQA_AGENT_VARIANT", raising=False)
    default = RepoEnv(TASK, PROFILE)                      # lean since 2026-09-19: no overview, structural tree map
    assert [s["name"] for s in default.specs()] == ["find_symbol", "grep", "read_file", "list_dir"]
    assert "Repository map" in default.initial_messages()[1].content
    full = RepoEnv(TASK, PROFILE, variant="full")
    assert [s["name"] for s in full.specs()] == ["overview", "find_symbol", "grep", "read_file", "list_dir"]
    b = RepoEnv(TASK, PROFILE, variant="bash")
    assert [s["name"] for s in b.specs()] == ["bash"] and "nl -ba" in b.initial_messages()[0].content
    assert "Repository map" in b.initial_messages()[1].content
    n = RepoEnv(TASK, PROFILE, variant="noindex")
    assert [s["name"] for s in n.specs()] == ["grep", "read_file", "list_dir"]
    assert "Repository map" not in n.initial_messages()[1].content and "overview" not in n.initial_messages()[0].content
    monkeypatch.setenv("CODEQA_AGENT_VARIANT", "bash_nomap")
    e = RepoEnv(TASK, PROFILE)
    assert e.variant.name == "bash_nomap" and "Repository map" not in e.initial_messages()[1].content
    assert default.budget == b.budget == n.budget


@pytest.mark.skipif(not shell.sandbox_exec_works(), reason="sandbox-exec layer not active on this machine")
def test_sandbox_layer_denies_reads_outside_snapshot_even_without_precheck():
    """Bypass the pre-check and call the sandboxed shell directly: reading the project's .env.example must be denied,
    reading inside the snapshot must work, and writing inside it must fail."""
    import subprocess
    cwd = paths.repo_dir(REPO_ID)
    env = {"PATH": str(shell.allowlist_dir())}
    prof = shell._sandbox_profile(cwd)
    outside = subprocess.run([shell.SANDBOX_EXEC, "-p", prof, shell.BASH, "-r", "-c", f"cat {paths.ROOT / '.env.example'}"],
                             capture_output=True, text=True, env=env, cwd=cwd)
    assert outside.returncode != 0 and "TINKER" not in outside.stdout
    inside = subprocess.run([shell.SANDBOX_EXEC, "-p", prof, shell.BASH, "-r", "-c", "head -1 README.md"],
                            capture_output=True, text=True, env=env, cwd=cwd)
    assert inside.returncode == 0 and inside.stdout.strip()
    # `-r` forbids redirection, so use a binary that writes: sed -i is blocked by precheck but not by bash -r; the sandbox must stop it
    write = subprocess.run([shell.SANDBOX_EXEC, "-p", prof, shell.BASH, "-r", "-c", "sed -i '' 's/Flask/FLASK/' README.md"],
                           capture_output=True, text=True, env=env, cwd=cwd)
    assert write.returncode != 0
    assert "FLASK" not in (cwd / "README.md").read_text()[:200]
