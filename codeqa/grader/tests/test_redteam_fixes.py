"""Rules added after the reward red-team (docs/research/reward_redteam_2026-09-19.md): each test is one verified finding."""
from __future__ import annotations

import pytest

from codeqa.grader.citations import check_citations, grounded_only_by_tolerance
from codeqa.grader.efficiency import TOKENS_PER_CALL, context_tokens, efficiency, redundant_reads
from codeqa.grader.judge import parse_verdict
from codeqa.grader.verifiers import cites_required, is_trivial_literal, literal_score, mentions_path, path_score, predicted_symbols, symbol_score
from codeqa.shared.contracts import Budget, Message, Span, Trace, TraceStats

S = "src/miniapp/auth/session.py"
T = "src/miniapp/auth/tokens.py"
R = "src/miniapp/api/routes.py"


def test_b1_one_line_tolerance_after_grep_hit(repo):
    grep_hit = [Span(path=S, start=35, end=35)]
    rep = check_citations(f"[{S}:L35-L36]", grep_hit, repo.repo_id, repo=repo)
    assert rep.all_grounded
    assert grounded_only_by_tolerance(f"[{S}:L35-L36]", grep_hit, repo) == 1        # logged as a near miss
    assert not check_citations(f"[{S}:L35-L40]", grep_hit, repo.repo_id, repo=repo).all_grounded   # four lines past: still unread
    assert not check_citations(f"[{T}:L12-L13]", grep_hit, repo.repo_id, repo=repo).all_grounded   # other file: still unread


def test_b2_body_citation_counts_and_question_names_excluded(repo):
    rep = check_citations(f"[{S}:L36-L42]", [Span(path=S, start=30, end=48)], repo.repo_id, repo=repo)
    assert f"{S}:validate_session" in predicted_symbols("validate_session does the check", rep, repo)
    assert f"{S}:validate_session" not in predicted_symbols("validate_session does the check", rep, repo, exclude_names={"validate_session"})


def test_b3_call_site_citation_supports_gold_callees(repo):
    rep = check_citations(f"[{S}:L35-L42]", [Span(path=S, start=30, end=48)], repo.repo_id, repo=repo)
    gold = [f"{T}:verify_token", f"{S}:Session.is_expired"]
    s = symbol_score("validate_session calls verify_token and Session.is_expired", rep, gold, repo,
                     question="Which functions does `validate_session` call directly?")
    assert s == 1.0
    # naming a callee without any cited line containing it earns nothing
    rep2 = check_citations(f"[{S}:L45-L48]", [Span(path=S, start=45, end=48)], repo.repo_id, repo=repo)
    assert symbol_score("it calls verify_token", rep2, gold, repo, question="Which functions does `validate_session` call directly?") == 0.0


def test_b4_precision_guard_on_single_gold(repo):
    rep = check_citations(f"[{S}:L1-L48]", [Span(path=S, start=1, end=48)], repo.repo_id, repo=repo)
    shotgun = "SessionExpired Session is_expired create_session validate_session expire_session"
    assert symbol_score(shotgun, rep, [f"{S}:validate_session"], repo) == pytest.approx(0.5)
    assert symbol_score("validate_session", rep, [f"{S}:validate_session"], repo) == 1.0
    assert path_score({"a.py", "b.py", "c.py", "d.py"}, ["a.py"], repo) == pytest.approx(0.5)
    assert path_score({"a.py", "b.py"}, ["a.py"], repo) == 1.0


def test_b5_path_tasks_must_name_the_file():
    assert mentions_path("The handler lives in routes.py [x:L1]", R)
    assert mentions_path(f"See {R} for it", R)
    assert not mentions_path(f"No idea. [{R}:L1]", R)              # only inside the bracket


def test_b6_literals(repo):
    assert literal_score("the separator is '.'", ".") == 1.0
    assert is_trivial_literal("False") and is_trivial_literal("0") and not is_trivial_literal("change-me")
    rep = check_citations("[src/miniapp/config.py:L7-L7]", [Span(path="src/miniapp/config.py", start=1, end=27)], repo.repo_id, repo=repo)
    assert cites_required(rep, [Span(path="src/miniapp/config.py", start=7, end=7)], repo)
    assert not cites_required(rep, [Span(path="src/miniapp/config.py", start=20, end=25)], repo)


def test_b7_judge_string_booleans():
    v = parse_verdict('{"items": [{"id": 1, "satisfied": "false"}, {"id": 2, "satisfied": "True"}, {"id": 3, "satisfied": true}]}', 3)
    assert v.satisfied == [False, True, True]


def test_b9_grep_spans_are_not_redundant_reads():
    reads = [Span(path="a", start=30, end=48)] + [Span(path="a", start=n, end=n) for n in range(31, 41)]
    assert redundant_reads(reads) == 0
    assert redundant_reads([Span(path="a", start=30, end=48), Span(path="a", start=35, end=40)]) == 1


def test_b8_final_context_accounting():
    msgs = [Message(role="user", content="q"),
            Message(role="assistant", content="", usage={"prompt_tokens": 4000, "completion_tokens": 200}),
            Message(role="tool", content="x"),
            Message(role="assistant", content="ans", usage={"prompt_tokens": 9000, "completion_tokens": 300})]
    tr = Trace(task_id="t", profile="p", messages=msgs, answer="ans", stats=TraceStats(tool_calls=1, prompt_tokens=13000))
    assert context_tokens(tr) == (4000, 9300)
    b = Budget(max_tool_calls=12)
    eff, over = efficiency(tr.stats, b, "multiplicative", prefix_tokens=4000, final_tokens=9300)
    assert eff == 1.0 and not over                                    # 9300 / (4000 + 12*1500) = 0.42: free zone
    eff_old, _ = efficiency(tr.stats, b, "multiplicative")            # cumulative fallback over-counts
    assert eff_old < eff
