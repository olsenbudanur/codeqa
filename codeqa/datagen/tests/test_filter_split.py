from codeqa.datagen import filter as flt
from codeqa.datagen.split import _table
from codeqa.shared.contracts import Task


def _t(i, q, repo="a__b__1234567", src="structural", tt="locate"):
    return Task(task_id=f"t{i}", repo_id=repo, question=q, task_type=tt, source=src)


def test_dedupe_by_token_overlap_within_repo():
    ts = [_t(1, "Where is the session cookie signed and verified?"),
          _t(2, "Where is the session cookie verified and signed?"),          # same tokens -> dup
          _t(3, "Where is the session cookie signed and verified?", repo="c__d__1234567"),  # other repo -> kept
          _t(4, "Which classes extend BaseGlobalPooling directly?")]
    kept, dropped = flt.dedupe(ts)
    assert [t.task_id for t in kept] == ["t1", "t3", "t4"] and dropped == [("t2", "t1")]


def test_dedupe_keeps_templated_questions_with_different_identifiers():
    ts = [_t(1, "Which files import the `pkg.a` module?"), _t(2, "Which files import the `pkg.b` module?"),
          _t(3, "What is the default value of the `x` parameter of `Foo.run`?"), _t(4, "What is the default value of the `x` parameter of `Bar.run`?")]
    kept, dropped = flt.dedupe(ts)
    assert len(kept) == 4 and dropped == []


def test_difficulty_score_uses_lenient_then_found():
    assert flt.difficulty_score({"n_answered": 2, "correct_lenient": 0.5, "found": 1.0}) == 0.5
    assert flt.difficulty_score({"n_answered": 1, "correct_lenient": 1.0, "found": 0.25}) == 0.25
    assert flt.difficulty_score({"n_answered": 0, "found": None}) is None


def test_lenient_correctness_and_found():
    from codeqa.shared.contracts import Grading, Span
    t = Task(task_id="x", repo_id="a__b__1234567", question="q", task_type="locate", source="structural",
             grading=Grading(expected_paths=["src/p/retrieval_rag.py"], expected_symbols=["src/p/retrieval_rag.py:CustomHFIndex"],
                             required_citations=[Span(path="src/p/retrieval_rag.py", start=100, end=140)]))
    assert flt.lenient_correctness(t, "It is the CustomHFIndex class in retrieval_rag.py") == 1.0
    assert flt.lenient_correctness(t, "Somewhere in the rag module") == 0.0
    assert flt.found_gold(t, [Span(path="src/p/retrieval_rag.py", start=130, end=200)]) is True
    assert flt.found_gold(t, [Span(path="src/p/retrieval_rag.py", start=1, end=99)]) is False
    v = Task(task_id="y", repo_id="a__b__1234567", question="q", task_type="value", source="structural", grading=Grading(expected_literal="utf-8", expected_paths=["a.py"]))
    assert flt.lenient_correctness(v, "The default is `utf-8`.") == 1.0


def test_table_counts_programmatic_share():
    ts = [_t(1, "a", src="codescout", tt="locate"), _t(2, "b", src="teacher", tt="explain"), _t(3, "c", src="deepcodebench", tt="enumerate")]
    tab = _table(ts)
    assert tab["total"] == 3 and tab["rows"]["teacher"]["explain"] == 1 and abs(tab["programmatic_share"] - 2 / 3) < 1e-3


def test_merge_records_weights_by_samples():
    a = {"n": 2, "errors": 0, "answered": 0.5, "format_ok": 0.0, "grounded": 0.0, "found": 1.0, "correct": 0.0, "correct_lenient_all": 0.5,
         "reward": 0.0, "tool_calls": 6.0, "n_answered": 1, "correct_lenient": 1.0, "n_format_ok": 0, "correct_given_format": None, "stops": {"answer": 1, "budget": 1}}
    b = {"n": 2, "errors": 0, "answered": 1.0, "format_ok": 0.5, "grounded": 0.5, "found": 0.5, "correct": 0.5, "correct_lenient_all": 0.5,
         "reward": 0.5, "tool_calls": 4.0, "n_answered": 2, "correct_lenient": 0.5, "n_format_ok": 1, "correct_given_format": 1.0, "stops": {"answer": 2}}
    m = flt.merge_records(a, b)
    assert m["n"] == 4 and m["answered"] == 0.75 and m["found"] == 0.75 and m["tool_calls"] == 5.0
    assert m["n_answered"] == 3 and abs(m["correct_lenient"] - (1.0 * 1 + 0.5 * 2) / 3) < 1e-3
    assert m["n_format_ok"] == 1 and m["correct_given_format"] == 1.0 and m["stops"] == {"answer": 3, "budget": 1}


def test_strip_reference_only_for_non_teacher_sources():
    from codeqa.datagen.split import strip_reference
    from codeqa.shared.contracts import Grading
    d = Task(task_id="d", repo_id="a__b__1234567", question="q", task_type="explain", source="deepcodebench", grading=Grading(rubric=["f"], reference_answer="long"))
    t = Task(task_id="t", repo_id="a__b__1234567", question="q", task_type="explain", source="teacher", grading=Grading(rubric=["f"], reference_answer="short"))
    assert strip_reference(d).grading.reference_answer is None and strip_reference(d).grading.rubric == ["f"]
    assert strip_reference(t).grading.reference_answer is None            # teacher too: train files are rubric-only
