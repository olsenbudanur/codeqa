"""The snapshot's own manifest.json is index metadata, not a repository file: the bash tool must neither list it nor
read it (LOG 2026-09-20 18:30: Scholia cited `[manifest.json:L4]` and the grader called it fabricated)."""
from __future__ import annotations

from codeqa.agent import shell


def test_metadata_precheck_refuses_direct_reads_only_at_the_root() -> None:
    m = ["manifest.json"]
    assert shell.metadata_precheck("cat manifest.json", m)
    assert shell.metadata_precheck("head -100 ./manifest.json", m)
    assert shell.metadata_precheck("nl -ba manifest.json | sed -n '1,5p'", m)
    assert shell.metadata_precheck("cat package/manifest.json", m) is None      # a real file elsewhere
    assert shell.metadata_precheck("grep -rn manifest .", m) is None            # the word, not the file
    assert shell.metadata_precheck("cat manifest.json", []) is None             # the repo really has one at its root


def test_hide_metadata_drops_listing_and_grep_lines_for_the_root_file_only() -> None:
    out = "\n".join([
        "total 108",
        "-rw-r--r-- 1 u u 340 Sep 20 AGENTS.md",
        "-rw-r--r-- 1 u u 2000 Sep 20 manifest.json",
        "./manifest.json",
        "./src/manifest.json",
        'manifest.json:4: "url": "https://github.com/x/y"',
        "README.md",
    ])
    kept = shell.hide_metadata(out, ["manifest.json"]).split("\n")
    assert kept == ["total 108", "-rw-r--r-- 1 u u 340 Sep 20 AGENTS.md", "./src/manifest.json", "README.md"]
    assert shell.hide_metadata(out, []) == out
