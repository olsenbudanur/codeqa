from codeqa.datagen import derive
from codeqa.datagen.resolve import RepoIndex
from codeqa.datagen.sources import rewrite
from codeqa.shared.contracts import FileEntry, IndexSymbol, Manifest


def _idx() -> RepoIndex:
    m = Manifest(repo_id="acme__spec__abc1234", url="u", sha="0" * 40, files=[
        FileEntry(path="spectree/utils.py", lang="python", lines=100, bytes=1),
        FileEntry(path="spectree/spec.py", lang="python", lines=100, bytes=1),
    ])
    syms = [IndexSymbol(name="parse_params", kind="function", path="spectree/utils.py", start=10, end=30),
            IndexSymbol(name="SpecTree", kind="class", path="spectree/spec.py", start=5, end=90),
            IndexSymbol(name="validate", kind="method", path="spectree/spec.py", start=40, end=60, parent="SpecTree")]
    return RepoIndex(m, syms)


def test_resolve_gold_requires_every_file_and_one_entity_per_file():
    idx = _idx()
    ok = derive.resolve_gold(idx, ["spectree/utils.py", "spectree/spec.py"],
                             ["spectree/utils.py:parse_params", "spectree/spec.py:SpecTree", "spectree/spec.py:SpecTree.validate", "spectree/spec.py:SpecTree.gone"])
    assert ok is not None
    paths, syms, spans = ok
    assert paths == ["spectree/spec.py", "spectree/utils.py"]
    assert syms == ["spectree/spec.py:SpecTree.validate", "spectree/utils.py:parse_params"]
    assert {(s.path, s.start) for s in spans} == {("spectree/utils.py", 10), ("spectree/spec.py", 40)}
    assert derive.resolve_gold(idx, ["spectree/missing.py"], ["spectree/missing.py:f"]) is None
    assert derive.resolve_gold(idx, ["spectree/utils.py"], ["spectree/utils.py:renamed"]) is None


def test_select_repos_excludes_and_picks_newest_commit():
    def row(repo, num, sha, ents=("pkg/a.py:f",)):
        return {"repo": repo, "instance_id": f"{repo.replace('/', '__')}-{num}", "base_commit": sha, "problem_statement": "x",
                "file_changes": [{"file": "pkg/a.py", "changes": {"edited_entities": list(ents), "edited_modules": None}}]}
    rows = [row("a/x", 1, "s1"), row("a/x", 30, "s30"), row("a/x", 7, "s7"), row("b/y", 2, "t2"), row("c/z", 9, "u9"), row("c/z", 10, "u10", ents=())]
    sel = derive.select_repos(rows, n=2, exclude={"b/y"})
    assert sel == [("a/x", 3, "s30"), ("c/z", 1, "u9")]


def test_rewrite_leak_detection():
    terms = rewrite.forbidden_terms(["sqlglot/dialects/snowflake.py"], ["sqlglot/dialects/snowflake.py:Snowflake.Parser._parse_div0", "geopandas/base.py:GeoPandasBase.explode"])
    assert rewrite.leaks("Where is Snowflake's DIV0 function converted to an IFF expression?", terms) is None      # domain word ok
    assert rewrite.leaks("Where is the current explode() method for GeoDataFrame objects?", terms) == "explode"     # code-styled
    assert rewrite.leaks("Where is `explode` implemented?", terms) == "explode"
    assert rewrite.leaks("Where does _parse_div0 handle it?", terms) == "_parse_div0"
    assert rewrite.leaks("Which class in sqlglot/dialects/snowflake.py handles it?", terms) == "sqlglot/dialects/snowflake.py"
    assert rewrite.leaks("Where does GeoPandasBase split multi-part geometries?", terms) == "GeoPandasBase"
    assert rewrite.leaks("Where are multi-part geometries exploded into rows?", terms) is None
