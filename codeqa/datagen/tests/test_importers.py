from codeqa.datagen.resolve import RepoIndex
from codeqa.datagen.sources import import_deepcodebench as dcb
from codeqa.datagen.sources import import_sweqa as sq
from codeqa.shared.contracts import FileEntry, IndexSymbol, Manifest, Task


def _idx(repo_id: str) -> RepoIndex:
    m = Manifest(repo_id=repo_id, url="u", sha="0" * 40, files=[
        FileEntry(path="src/transformers/pipelines/base.py", lang="python", lines=300, bytes=1),
        FileEntry(path="src/flask/json/provider.py", lang="python", lines=200, bytes=1),
    ])
    syms = [
        IndexSymbol(name="pad_collate_fn", kind="function", path="src/transformers/pipelines/base.py", start=50, end=120),
        IndexSymbol(name="inner", kind="function", path="src/transformers/pipelines/base.py", start=70, end=110),
        IndexSymbol(name="DefaultJSONProvider", kind="class", path="src/flask/json/provider.py", start=100, end=190),
    ]
    return RepoIndex(m, syms)


DCB_ROW = {
    "id": "0fbd5b3c-87d4-4894-9319-d979c2f45997",
    "question": "Under what condition does the collate function use the feature extractor's padding value?",
    "answer": "In `pad_collate_fn.inner` (src/transformers/pipelines/base.py) it checks `tokenizer is None`. See also missing/file.py.",
    "facts": ["pad_collate_fn.inner is defined in src/transformers/pipelines/base.py.", " "],
    "metadata": {"repo": "https://github.com/huggingface/transformers.git", "commit": "a1ad9197c5756858e9014a0e01fe5fb1791efdf2"},
}


def test_deepcodebench_without_index_keeps_raw_paths():
    t = dcb.to_task(DCB_ROW)
    assert t.repo_id == "huggingface__transformers__a1ad919" and t.split == "train" and t.source == "deepcodebench"
    assert t.grading.expected_paths == ["missing/file.py", "src/transformers/pipelines/base.py"]
    assert t.grading.rubric == ["pad_collate_fn.inner is defined in src/transformers/pipelines/base.py."]
    assert t.grading.required_citations == [] and t.task_type == "explain"


def test_deepcodebench_with_index_resolves_paths_and_symbols():
    t = dcb.to_task(DCB_ROW, "eval", index=_idx("huggingface__transformers__a1ad919"))
    assert t.split == "eval"
    assert t.grading.expected_paths == ["src/transformers/pipelines/base.py"]          # missing path dropped
    c = t.grading.required_citations
    assert c and c[0].path == "src/transformers/pipelines/base.py" and (c[0].start, c[0].end) == (70, 110)
    Task.model_validate_json(t.model_dump_json())


def test_sweqa_citations_resolve_basenames_and_clip():
    idx = _idx("pallets__flask__85c5d93")
    row = {"question": "Where is ensure_ascii applied?",
           "answer": "`DefaultJSONProvider` in provider.py sets it (lines 144-148) and again at line 250. In json/provider.py line 166 too."}
    t = sq.to_task(row, "flask", "85c5d93" + "0" * 33, 7, index=idx)
    assert t.task_id == "sweqa-flask-007" and t.split == "eval" and t.task_type == "locate"
    spans = {(s.path, s.start, s.end) for s in t.grading.required_citations}
    assert spans == {("src/flask/json/provider.py", 144, 148), ("src/flask/json/provider.py", 166, 166)}   # line 250 past EOF dropped
    assert t.grading.expected_paths == ["src/flask/json/provider.py"]


def test_parse_repo_commits_handles_blank_and_partial_lines():
    text = "https://github.com/pallets/flask 85c5d93\n\nhttps://github.com/reflex-dev/\nhttps://github.com/scikit-learn/scikit-learn adb1ae7\n"
    assert sq.parse_repo_commits(text) == {"flask": "85c5d93", "scikit_learn": "adb1ae7"}


def test_deepcodebench_review_fixes_apply_at_import():
    row = dict(DCB_ROW, id="4344b2a4-42dd-441f-8c9c-2438db99176b", answer="range ≥ 2024.1.1 and < 2025.3.0", facts=["corresponds to ≥ 2024.1.1"])
    t = dcb.to_task(row)
    assert "2024.12.1" in t.grading.reference_answer and t.grading.rubric == ["corresponds to ≥ 2024.12.1"]
    row = dict(DCB_ROW, id="d806e6bd-9e8a-4759-8b83-01fdfdfba005", question="What are the two distinct error conditions checked?")
    assert dcb.to_task(row).task_type == "explain"          # heuristic would say enumerate
