from codeqa.grader.citations import check_citations
from codeqa.grader.verifiers import f1, literal_score, path_score, predicted_symbols, symbol_score, uses_judge
from codeqa.shared.contracts import Span

R = "src/miniapp/api/routes.py"


def test_literal_whole_token_and_normalization():
    assert literal_score("The default is 30 seconds", "30") == 1.0
    assert literal_score("The default is 300 seconds", "30") == 0.0
    assert literal_score("timeout `30`.", "'30'") == 1.0
    assert literal_score("uses change-me as secret", "change-me") == 1.0


def test_path_score_single_exact_multi_f1(repo):
    assert path_score({"a.py"}, ["a.py"], repo) == 1.0
    assert path_score({"a.py", "b.py"}, ["a.py"], repo) == 1.0
    assert path_score({"b.py"}, ["a.py"], repo) == 0.0
    assert abs(path_score({"a.py", "c.py"}, ["a.py", "b.py"], repo) - 0.5) < 1e-9


def test_f1_edge_cases():
    assert f1(set(), {"a"}) == 0.0 and f1({"a"}, set()) == 0.0 and f1({"a"}, {"a"}) == 1.0


def test_symbol_prediction_needs_name_and_cited_definition(repo):
    reads = [Span(path=R, start=1, end=39)]
    named_and_cited = check_citations(f"HealthRoute and UsersRoute [{R}:L1-L39]", reads, repo.repo_id, repo=repo)
    assert predicted_symbols("HealthRoute and UsersRoute", named_and_cited, repo) >= {f"{R}:HealthRoute", f"{R}:UsersRoute"}
    wrong_lines = check_citations(f"HealthRoute [{R}:L27-L39]", reads, repo.repo_id, repo=repo)
    assert f"{R}:HealthRoute" not in predicted_symbols("HealthRoute", wrong_lines, repo)
    assert predicted_symbols("nothing named", named_and_cited, repo) == set()


def test_symbol_score_any_of_vs_f1(repo):
    reads = [Span(path=R, start=1, end=39)]
    rep = check_citations(f"[{R}:L1-L39]", reads, repo.repo_id, repo=repo)
    assert symbol_score("HealthRoute", rep, [f"{R}:HealthRoute"], repo) == 1.0
    assert symbol_score("HealthRoute only", rep, [f"{R}:HealthRoute", f"{R}:UsersRoute"], repo) < 1.0
    assert symbol_score("HealthRoute and UsersRoute", rep, [f"{R}:HealthRoute", f"{R}:UsersRoute"], repo) == 1.0
    # naming an extra non-gold symbol lowers precision
    assert symbol_score("HealthRoute, UsersRoute and BaseRoute", rep, [f"{R}:HealthRoute", f"{R}:UsersRoute"], repo) < 1.0


def test_uses_judge_only_for_judged_types_with_material(tasks):
    assert uses_judge(tasks["mini-trace"]) and uses_judge(tasks["mini-explain"])
    assert not uses_judge(tasks["mini-locate"]) and not uses_judge(tasks["mini-enumerate"])


def test_unverifiable_task_with_rubric_falls_back_to_judge(tasks):
    t = tasks["mini-enumerate"].model_copy(deep=True)
    t.grading.expected_symbols, t.grading.expected_paths = [], []
    assert not uses_judge(t)                                  # no gold, no rubric: nothing to grade
    t.grading.rubric = ["names `HealthRoute`"]
    assert uses_judge(t)                                      # no gold but a rubric: judge
    t.grading.expected_paths = ["src/miniapp/api/routes.py"]
    assert not uses_judge(t)                                  # gold present: verify, ignore the rubric
