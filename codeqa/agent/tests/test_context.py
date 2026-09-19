import pytest

from codeqa.agent.env import RepoEnv
from codeqa.agent.indexing.repomap import count_tokens
from codeqa.agent.variants import VARIANTS, resolve
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Task

FLASK = "pallets__flask__85c5d93"
pytestmark = pytest.mark.skipif(not (paths.repo_dir(FLASK) / "manifest.json").exists(), reason="flask snapshot not present")
PROFILE = EndpointProfile(name="t", kind="anthropic", model="x")


def _task():
    return Task(task_id="t1", repo_id=FLASK, question="Where is the session cookie signed?", task_type="locate", source="teacher")


def test_default_is_lean(monkeypatch):
    monkeypatch.delenv("CODEQA_AGENT_VARIANT", raising=False)
    env = RepoEnv(_task(), PROFILE)
    assert env.variant.name == "default" and env.variant.map == "tree" and not env.variant.needs_summaries
    assert [t.name for t in env.tools()] == ["find_symbol", "grep", "read_file", "list_dir"]
    assert [s["name"] for s in env.specs()] == ["find_symbol", "grep", "read_file", "list_dir"]
    sysm, user = env.initial_messages()
    assert "overview" not in sysm.content and "repository map" in sysm.content
    assert "Repository map" in user.content and 0 < count_tokens(env.repo_map) <= 1000
    assert "src/" in env.repo_map and "sessions.py" in env.repo_map


def test_full_variant_restores_day_one_design():
    env = RepoEnv(_task(), PROFILE, variant="full")
    assert [t.name for t in env.tools()] == ["overview", "find_symbol", "grep", "read_file", "list_dir"]
    assert "Prefer overview and find_symbol" in env.initial_messages()[0].content
    assert count_tokens(env.repo_map) > 1000 and env.variant.needs_summaries


def test_nomap_variant_has_no_map():
    env = RepoEnv(_task(), PROFILE, variant="nomap")
    sysm, user = env.initial_messages()
    assert env.repo_map == "" and user.content.startswith("Repository: ") and "map below" not in sysm.content
    assert len(env.tools()) == 4


def test_env_var_selects_variant(monkeypatch):
    monkeypatch.setenv("CODEQA_AGENT_VARIANT", "full")
    assert resolve(None).name == "full"
    monkeypatch.setenv("CODEQA_AGENT_VARIANT", "bogus")
    with pytest.raises(KeyError):
        resolve(None)
    assert set(VARIANTS) >= {"default", "lean", "full", "nomap", "noindex", "bash"}
