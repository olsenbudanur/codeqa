"""Structural index with tree-sitter: symbols.json (C2). Python first; other languages via the same pack later."""
from __future__ import annotations

from pathlib import Path

from tree_sitter_language_pack import get_parser

from codeqa.shared import paths
from codeqa.shared.contracts import IndexSymbol, Manifest
from codeqa.shared.jsonl import dump_json

PY_DEF_TYPES = {"function_definition": "function", "class_definition": "class"}


def _signature(src: bytes, node) -> str:
    first = src[node.start_byte:node.end_byte].split(b"\n", 1)[0].decode("utf-8", "replace")
    return first.strip()[:200]


def python_symbols(rel_path: str, src: bytes) -> list[IndexSymbol]:
    parser = get_parser("python")
    tree = parser.parse(src)
    out: list[IndexSymbol] = []

    def walk(node, parent_class: str | None):
        for child in node.children:
            kind = PY_DEF_TYPES.get(child.type)
            if kind:
                name_node = child.child_by_field_name("name")
                name = src[name_node.start_byte:name_node.end_byte].decode() if name_node else "?"
                sym_kind = "method" if (kind == "function" and parent_class) else kind
                out.append(IndexSymbol(name=name, kind=sym_kind, path=rel_path, start=child.start_point[0] + 1,
                                       end=child.end_point[0] + 1, parent=parent_class, signature=_signature(src, child)))
                body = child.child_by_field_name("body")
                if body is not None:
                    walk(body, name if kind == "class" else parent_class)
            elif child.type == "decorated_definition":
                walk(child, parent_class)
            elif child.type in ("block", "module", "if_statement", "try_statement", "else_clause", "except_clause"):
                walk(child, parent_class)
    walk(tree.root_node, None)
    return out


def build_index(manifest: Manifest) -> list[IndexSymbol]:
    root = paths.repo_dir(manifest.repo_id)
    symbols: list[IndexSymbol] = []
    for f in manifest.files:
        if f.lang != "python":
            continue
        src = (root / f.path).read_bytes()
        try:
            symbols.extend(python_symbols(f.path, src))
        except Exception:
            continue
    out = paths.index_dir(manifest.repo_id) / "symbols.json"
    dump_json(out, {"repo_id": manifest.repo_id, "symbols": [s.model_dump() for s in symbols]})
    return symbols


_SYMBOL_CACHE: dict[tuple[str, float], list[IndexSymbol]] = {}


def load_symbols(repo_id: str) -> list[IndexSymbol]:
    """Cached per process by (repo_id, file mtime): 128 concurrent envs on one repo share one parsed index."""
    import json
    p = paths.index_dir(repo_id) / "symbols.json"
    key = (repo_id, p.stat().st_mtime)
    if key not in _SYMBOL_CACHE:
        data = json.loads(p.read_text())
        _SYMBOL_CACHE.clear() if len(_SYMBOL_CACHE) > 64 else None
        _SYMBOL_CACHE[key] = [IndexSymbol.model_validate(s) for s in data["symbols"]]
    return _SYMBOL_CACHE[key]
