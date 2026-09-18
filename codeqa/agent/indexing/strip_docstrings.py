"""`<repo_id>__nodoc`: a copy of the snapshot with every Python docstring blanked, plus its own index.

Why: a locate question paraphrased from a docstring is trivially grep-able. Structural tasks are generated
against this variant (gap_specs §10). Line numbers are preserved (each docstring becomes "" + the same
number of newlines) so citations transfer between the two variants.
"""
from __future__ import annotations

import shutil

from tree_sitter_language_pack import get_parser

from codeqa.agent.indexing.index import build_index
from codeqa.shared import paths
from codeqa.shared.contracts import FileEntry, IndexSymbol, Manifest
from codeqa.shared.jsonl import dump_json

BODY_TYPES = {"module", "block"}


def strip_python_docstrings(src: bytes) -> tuple[bytes, int]:
    """Blank docstrings (first string-expression statement of a module/class/function body). Returns (src, count)."""
    parser = get_parser("python")
    tree = parser.parse(src)
    spans: list[tuple[int, int]] = []

    def first_stmt_docstring(body) -> None:
        for child in body.children:
            if child.type == "comment":
                continue
            # newer grammars put the docstring `string` directly in the body; older ones wrap it in expression_statement
            s = child if child.type == "string" else (
                child.children[0] if child.type == "expression_statement" and len(child.children) == 1 and child.children[0].type == "string" else None)
            if s is not None:
                spans.append((s.start_byte, s.end_byte))
            return

    def walk(node) -> None:
        if node.type in BODY_TYPES:
            first_stmt_docstring(node)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    if not spans:
        return src, 0
    out = bytearray()
    pos = 0
    for start, end in sorted(spans):
        out += src[pos:start]
        out += b'""' + b"\n" * src[start:end].count(b"\n")
        pos = end
    out += src[pos:]
    return bytes(out), len(spans)


def make_nodoc(manifest: Manifest, force: bool = False) -> tuple[Manifest, list[IndexSymbol]]:
    src_root = paths.repo_dir(manifest.repo_id)
    nodoc_id = manifest.repo_id + "__nodoc"
    dest = paths.repo_dir(nodoc_id)
    mpath = dest / "manifest.json"
    if mpath.exists() and not force:
        m = Manifest.model_validate_json(mpath.read_text())
        from codeqa.agent.indexing.index import load_symbols
        try:
            return m, load_symbols(nodoc_id)
        except FileNotFoundError:
            return m, build_index(m)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src_root, dest, ignore=shutil.ignore_patterns("manifest.json"))
    files: list[FileEntry] = []
    stripped = 0
    for f in manifest.files:
        p = dest / f.path
        if f.lang == "python" and p.is_file():
            new, n = strip_python_docstrings(p.read_bytes())
            if n:
                p.write_bytes(new)
                stripped += n
            files.append(FileEntry(path=f.path, lang=f.lang, lines=f.lines, bytes=len(new)))
        else:
            files.append(f)
    m = Manifest(repo_id=nodoc_id, url=manifest.url, sha=manifest.sha, files=files,
                 dropped={**manifest.dropped, "docstrings_blanked": stripped})
    dump_json(mpath, m)
    return m, build_index(m)
