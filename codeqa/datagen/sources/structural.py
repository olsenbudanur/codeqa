"""B4: programmatic tasks from a repo's source + index (no upstream dataset).

  locate     docstring (paraphrased by Haiku, name withheld)          -> the symbol's path + qualified name + span
  value      module constant / default argument                        -> literal
  enumerate  importers of a module / direct subclasses of a class      -> path set / symbol set
  trace      repo-defined functions a function calls directly          -> symbol set

Facts are read from the ORIGINAL snapshot (docstrings intact); tasks point at `<repo_id>__nodoc` when it exists, whose
line numbers are identical, so a locate question cannot be solved by grepping the docstring.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from tree_sitter_language_pack import get_parser

from codeqa.shared import paths
from codeqa.shared.contracts import Grading, IndexSymbol, Manifest, Span, Task

SKIP_DIR_PARTS = {"tests", "test", "testing", "docs", "doc", "examples", "example", "benchmarks", "benchmark", "scripts", "conftest", "migrations", "vendor", "_vendor", "third_party", "build"}
SKIP_FILE_RE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test|conftest|setup|__main__|version|_version)\.py$")
LITERAL_TYPES = {"integer", "float", "string", "true", "false", "none"}
GENERIC_NAMES = {"main", "run", "get", "set", "init", "call", "apply", "update", "build", "create", "load", "save", "parse", "format", "close", "open", "read", "write", "start", "stop", "reset", "setup", "check", "validate", "process", "handle", "execute", "compute", "forward", "fit", "predict", "transform", "wrapper", "inner", "decorator", "helper", "func", "function", "method", "__init__", "__call__", "__repr__", "__str__", "__eq__", "__hash__", "__len__", "__iter__", "__getitem__", "__setitem__", "__enter__", "__exit__"}
TEMPLATES = {
    "value_const": ["What value is the constant `{name}` set to in this codebase?",
                    "What is the module-level constant `{name}` defined as?",
                    "Which literal value does `{name}` hold where it is defined?"],
    "value_default": ["What is the default value of the `{param}` parameter of `{qual}`{where}?",
                      "If `{qual}`{where} is called without `{param}`, what value does `{param}` take?",
                      "What default does `{qual}`{where} use for its `{param}` argument?"],
    "enum_importers": ["Which files import the `{dotted}` module?",
                       "Which modules in this repository import directly from `{dotted}`?",
                       "List the files that import `{dotted}`."],
    "enum_subclasses": ["Which classes directly inherit from `{base}`?",
                        "What are the direct subclasses of `{base}` in this codebase?",
                        "Which classes extend `{base}` directly?"],
    "trace": ["Which functions defined in this repository does `{qual}`{where} call directly?",
              "What repository-defined functions or methods are invoked inside `{qual}`{where}?",
              "List the direct callees of `{qual}`{where} that are defined in this codebase."],
}


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def norm_literal(src: bytes, node) -> str | None:
    """Literal node -> normalized string ('utf-8', '300', '0.5', 'True', 'None', '-1'); None if not a simple literal."""
    t = node.type
    if t == "unary_operator" and len(node.children) == 2 and node.children[1].type in ("integer", "float"):
        return _text(src, node.children[0]) + _text(src, node.children[1])
    if t not in LITERAL_TYPES:
        return None
    raw = _text(src, node)
    if t == "string":
        if node.child_count > 3 or raw[:1] in "fF" or raw[:2].lower() in ("rf", "fr", "bf"):
            return None                           # f-strings / concatenations / odd prefixes
        m = re.match(r"^[rRbBuU]*(['\"]{1,3})(.*)\1$", raw, re.S)
        if not m:
            return None
        val = m.group(2)
        return val if 0 < len(val) <= 60 and "\n" not in val else None
    return raw if len(raw) <= 30 else None


def docstring_of(src: bytes, body) -> str | None:
    if body is None:
        return None
    for child in body.children:
        if child.type == "comment":
            continue
        s = child if child.type == "string" else (
            child.children[0] if child.type == "expression_statement" and len(child.children) == 1 and child.children[0].type == "string" else None)
        if s is None:
            return None
        raw = _text(src, s)
        m = re.match(r"^[rRbBuU]*(['\"]{3}|['\"])(.*)\1$", raw, re.S)
        text = m.group(2) if m else raw
        return " ".join(text.split())
    return None


def name_tokens(*names: str) -> set[str]:
    """Words that would leak a symbol: the names themselves plus their snake_case / CamelCase parts (>= 4 chars)."""
    toks: set[str] = set()
    for n in names:
        if not n:
            continue
        toks.add(n.lower())
        for dotted in n.split("."):
            toks.add(dotted.lower())
        for part in re.split(r"[_.]", n):
            for w in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", part):
                if len(w) >= 4:
                    toks.add(w.lower())
    return toks


def contains_leak(question: str, toks: set[str]) -> str | None:
    words = set(re.findall(r"[a-z0-9_]+", question.lower()))
    q = question.lower()
    for t in toks:
        if t in words or ("_" in t and t in q) or ("." in t and t in q):
            return t
    return None


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class FuncFacts:
    sym: IndexSymbol
    qual: str
    docstring: str | None
    defaults: list[tuple[str, str]]                 # (param, literal)
    callees: list[str] = field(default_factory=list)  # resolved 'path:Qual'


@dataclass
class RepoFacts:
    repo_id: str
    funcs: list[FuncFacts]
    classes: list[tuple[IndexSymbol, str | None, list[str]]]      # (sym, docstring, base names)
    constants: dict[str, list[tuple[str, int, str]]]              # NAME -> [(path, line, literal)]
    importers: dict[str, list[tuple[str, int]]]                   # module path -> [(importer path, line)]
    module_of_path: dict[str, str]                                # path -> dotted module


def _is_skipped(path: str) -> bool:
    parts = path.split("/")
    return any(p in SKIP_DIR_PARTS for p in parts[:-1]) or bool(SKIP_FILE_RE.search(path))


def _dotted_variants(path: str) -> list[str]:
    mod = path[:-3] if path.endswith(".py") else path
    if mod.endswith("/__init__"):
        mod = mod[: -len("/__init__")]
    parts = mod.split("/")
    out = [".".join(parts)]
    for i in range(1, min(3, len(parts))):          # src/flask/app.py -> flask.app ; python-package/lightgbm/basic.py -> lightgbm.basic
        out.append(".".join(parts[i:]))
    return [o for o in out if o]


def _canonical_module(path: str, all_paths: set[str]) -> str:
    """Dotted name as it is imported: drop leading directories that are not packages (no __init__.py)."""
    mod = path[:-3] if path.endswith(".py") else path
    if mod.endswith("/__init__"):
        mod = mod[: -len("/__init__")]
    parts = mod.split("/")
    i = 0
    while i < len(parts) - 1 and f"{'/'.join(parts[: i + 1])}/__init__.py" not in all_paths:
        i += 1
    return ".".join(parts[i:])


def build_facts(manifest: Manifest, symbols: list[IndexSymbol]) -> RepoFacts:
    root = paths.repo_dir(manifest.repo_id)
    parser = get_parser("python")
    py_files = [f.path for f in manifest.files if f.lang == "python" and not _is_skipped(f.path)]
    mod_to_path: dict[str, str] = {}
    module_of_path: dict[str, str] = {}
    all_paths = {f.path for f in manifest.files}
    for p in py_files:
        module_of_path[p] = _canonical_module(p, all_paths)
        for v in _dotted_variants(p):
            mod_to_path.setdefault(v, p)
    syms_by_file: dict[str, list[IndexSymbol]] = defaultdict(list)
    for s in symbols:
        syms_by_file[s.path].append(s)
    # uniqueness of (parent, name) across the repo, for callee resolution and unambiguous questions
    count_by_key: dict[tuple[str | None, str], int] = defaultdict(int)
    count_by_name: dict[str, int] = defaultdict(int)
    for s in symbols:
        count_by_key[(s.parent, s.name)] += 1
        count_by_name[s.name] += 1

    funcs: list[FuncFacts] = []
    classes: list[tuple[IndexSymbol, str | None, list[str]]] = []
    constants: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    importers: dict[str, list[tuple[str, int]]] = defaultdict(list)

    def resolve_module(dotted: str, from_path: str, level: int) -> str | None:
        if level:
            base = from_path.rsplit("/", 1)[0] if "/" in from_path else ""
            for _ in range(level - 1):
                base = base.rsplit("/", 1)[0] if "/" in base else ""
            rel = dotted.replace(".", "/") if dotted else ""
            cand = "/".join(x for x in (base, rel) if x)
            for c in (cand + ".py", cand + "/__init__.py"):
                if c in module_of_path:
                    return c
            return None
        return mod_to_path.get(dotted)

    for path in py_files:
        try:
            src = (root / path).read_bytes()
            tree = parser.parse(src)
        except Exception:
            continue
        rootn = tree.root_node
        by_pos = {(s.start, s.name): s for s in syms_by_file.get(path, [])}
        # imports: module aliases and imported names, both resolved to repo paths when possible
        alias_to_path: dict[str, str] = {}
        name_to_path: dict[str, str] = {}
        for node in rootn.children:
            if node.type == "import_statement":
                for ch in node.named_children:
                    if ch.type == "dotted_name":
                        tgt = resolve_module(_text(src, ch), path, 0)
                        if tgt:
                            alias_to_path[_text(src, ch).split(".")[-1]] = tgt
                            importers[tgt].append((path, node.start_point[0] + 1))
                    elif ch.type == "aliased_import":
                        nm = ch.child_by_field_name("name"); al = ch.child_by_field_name("alias")
                        tgt = resolve_module(_text(src, nm), path, 0) if nm is not None else None
                        if tgt:
                            alias_to_path[_text(src, al) if al is not None else _text(src, nm).split(".")[-1]] = tgt
                            importers[tgt].append((path, node.start_point[0] + 1))
            elif node.type == "import_from_statement":
                modn = node.child_by_field_name("module_name")
                if modn is None:
                    continue
                level = 0; dotted = ""
                if modn.type == "relative_import":
                    for ch in modn.children:
                        if ch.type == "import_prefix":
                            level = _text(src, ch).count(".")
                        elif ch.type == "dotted_name":
                            dotted = _text(src, ch)
                else:
                    dotted = _text(src, modn)
                tgt = resolve_module(dotted, path, level)
                if tgt and tgt != path:
                    importers[tgt].append((path, node.start_point[0] + 1))
                names = [ch for ch in node.children_by_field_name("name")]
                for ch in names:
                    nm = ch.child_by_field_name("name") if ch.type == "aliased_import" else ch
                    al = ch.child_by_field_name("alias") if ch.type == "aliased_import" else None
                    if nm is None:
                        continue
                    imported = _text(src, nm)
                    local = _text(src, al) if al is not None else imported
                    sub = resolve_module((dotted + "." if dotted else "") + imported, path, level)
                    if sub:
                        alias_to_path[local] = sub
                        if sub != path:
                            importers[sub].append((path, node.start_point[0] + 1))   # `from pkg import util` imports pkg/util.py
                    elif tgt:
                        name_to_path[local] = tgt
        # module constants
        for node in rootn.children:
            # this grammar version puts `assignment` directly under module; older ones wrap it in expression_statement
            a = node if node.type == "assignment" else (
                node.children[0] if node.type == "expression_statement" and node.children and node.children[0].type == "assignment" else None)
            if a is not None:
                left, right = a.child_by_field_name("left"), a.child_by_field_name("right")
                if left is not None and right is not None and left.type == "identifier":
                    nm = _text(src, left)
                    if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", nm) and not nm.startswith("__"):
                        lit = norm_literal(src, right)
                        if lit is not None and lit not in ("None", ""):
                            constants[nm].append((path, node.start_point[0] + 1, lit))

        # functions / classes
        def walk(node, parent_class: str | None):
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    name_node = child.child_by_field_name("name")
                    name = _text(src, name_node) if name_node is not None else "?"
                    sym = by_pos.get((child.start_point[0] + 1, name))
                    body = child.child_by_field_name("body")
                    doc = docstring_of(src, body)
                    if child.type == "class_definition":
                        bases: list[str] = []
                        sup = child.child_by_field_name("superclasses")
                        if sup is not None:
                            for b in sup.named_children:
                                if b.type == "identifier":
                                    bases.append(_text(src, b))
                                elif b.type == "attribute":
                                    at = b.child_by_field_name("attribute")
                                    if at is not None:
                                        bases.append(_text(src, at))
                        if sym is not None:
                            classes.append((sym, doc, bases))
                        if body is not None:
                            walk(body, name)
                    else:
                        defaults: list[tuple[str, str]] = []
                        params = child.child_by_field_name("parameters")
                        if params is not None:
                            for p in params.named_children:
                                if p.type in ("default_parameter", "typed_default_parameter"):
                                    pn, pv = p.child_by_field_name("name"), p.child_by_field_name("value")
                                    if pn is not None and pv is not None:
                                        lit = norm_literal(src, pv)
                                        if lit is not None and lit not in ("None", ""):
                                            defaults.append((_text(src, pn), lit))
                        callees: list[str] = []
                        if sym is not None and body is not None:
                            seen: set[str] = set()
                            stack = [body]
                            while stack:
                                n = stack.pop()
                                if n.type == "call":
                                    fn = n.child_by_field_name("function")
                                    tgt_path = None; cname = None
                                    if fn is not None and fn.type == "identifier":
                                        cname = _text(src, fn)
                                        if cname in alias_to_path or cname in name_to_path:
                                            tgt_path = name_to_path.get(cname) or alias_to_path.get(cname)
                                        elif any(s.name == cname and s.parent is None for s in syms_by_file.get(path, [])):
                                            tgt_path = path
                                    elif fn is not None and fn.type == "attribute":
                                        obj, at = fn.child_by_field_name("object"), fn.child_by_field_name("attribute")
                                        if obj is not None and at is not None:
                                            cname = _text(src, at)
                                            if obj.type == "identifier":
                                                o = _text(src, obj)
                                                if o in ("self", "cls") and parent_class:
                                                    tgt_path = path
                                                elif o in alias_to_path:
                                                    tgt_path = alias_to_path[o]
                                    if tgt_path and cname and cname != name and cname not in GENERIC_NAMES and len(cname) >= 4:
                                        cands = [s for s in syms_by_file.get(tgt_path, []) if s.name == cname and s.kind in ("function", "method")]
                                        if fn.type == "attribute" and _text(src, fn.child_by_field_name("object")) in ("self", "cls"):
                                            cands = [s for s in cands if s.parent == parent_class]
                                        elif tgt_path != path or fn.type == "identifier":
                                            cands = [s for s in cands if s.parent is None or (tgt_path != path and s.kind == "function")]
                                        if len(cands) == 1:
                                            c = cands[0]
                                            q = f"{c.parent}.{c.name}" if c.parent else c.name
                                            key = f"{c.path}:{q}"
                                            if key not in seen:
                                                seen.add(key); callees.append(key)
                                stack.extend(n.children)
                        if sym is not None:
                            qual = f"{parent_class}.{name}" if parent_class else name
                            funcs.append(FuncFacts(sym=sym, qual=qual, docstring=doc, defaults=defaults, callees=callees))
                        if body is not None:
                            walk(body, parent_class)
                elif child.type in ("decorated_definition", "block", "if_statement", "try_statement", "else_clause", "except_clause"):
                    walk(child, parent_class)
        walk(rootn, None)

    facts = RepoFacts(repo_id=manifest.repo_id, funcs=funcs, classes=classes, constants=dict(constants),
                      importers={k: sorted(set(v)) for k, v in importers.items()}, module_of_path=module_of_path)
    facts.count_by_key = count_by_key      # type: ignore[attr-defined]
    facts.count_by_name = count_by_name    # type: ignore[attr-defined]
    return facts


# ---------------------------------------------------------------------------------------------------------------------
def _tid(repo_id: str, kind: str, key: str) -> str:
    return f"st-{kind}-{hashlib.sha1(f'{repo_id}|{kind}|{key}'.encode()).hexdigest()[:8]}"


def _where(facts: RepoFacts, sym: IndexSymbol) -> str:
    unique = facts.count_by_key[(sym.parent, sym.name)] == 1   # type: ignore[attr-defined]
    return "" if unique else f" in `{sym.path}`"


def _span(sym: IndexSymbol) -> Span:
    from codeqa.datagen.resolve import symbol_span
    return symbol_span(sym)


def generate(manifest: Manifest, symbols: list[IndexSymbol], task_repo_id: str, per_type: int = 15, cap: int = 60,
             seed: str | None = None) -> tuple[list[Task], dict[str, Any]]:
    """-> (tasks, stats). locate tasks carry the docstring in grading.reference_answer and question='' until paraphrased."""
    facts = build_facts(manifest, symbols)
    rng = random.Random(seed or manifest.repo_id)
    out: dict[str, list[Task]] = {"locate": [], "value": [], "enumerate": [], "trace": []}
    stats: dict[str, Any] = {"candidates": {}}

    # locate: public functions/methods/classes with a real docstring, unique (parent, name)
    loc_cands = [f for f in facts.funcs if f.docstring and len(f.docstring) >= 60 and not f.sym.name.startswith("_")
                 and f.sym.name not in GENERIC_NAMES and facts.count_by_key[(f.sym.parent, f.sym.name)] == 1]  # type: ignore[attr-defined]
    loc_cands += [FuncFacts(sym=s, qual=s.name, docstring=d, defaults=[]) for s, d, _ in facts.classes
                  if d and len(d) >= 60 and not s.name.startswith("_") and facts.count_by_key[(None, s.name)] == 1]  # type: ignore[attr-defined]
    rng.shuffle(loc_cands)
    stats["candidates"]["locate"] = len(loc_cands)
    for f in loc_cands[: per_type * 3]:                         # extra so paraphrase drops still leave per_type
        out["locate"].append(Task(
            task_id=_tid(task_repo_id, "loc", f"{f.sym.path}:{f.qual}"), repo_id=task_repo_id, question="", task_type="locate",
            source="structural", source_id=f"docstring:{f.sym.path}:{f.qual}",
            grading=Grading(expected_paths=[f.sym.path], expected_symbols=[f"{f.sym.path}:{f.qual}"],
                            reference_answer=f.docstring[:600], required_citations=[_span(f.sym)])))

    # value: constants defined once with one value; default args of unique functions
    const_cands = [(n, v[0]) for n, v in facts.constants.items() if len({x[2] for x in v}) == 1 and len(v) == 1]
    def_cands = [(f, p, lit) for f in facts.funcs if not f.sym.name.startswith("_") and f.sym.name not in GENERIC_NAMES
                 for p, lit in f.defaults if facts.count_by_name[f.sym.name] <= 3]   # type: ignore[attr-defined]
    rng.shuffle(const_cands); rng.shuffle(def_cands)
    stats["candidates"]["value"] = len(const_cands) + len(def_cands)
    half = per_type // 2
    for n, (p, line, lit) in const_cands[:half]:
        out["value"].append(Task(
            task_id=_tid(task_repo_id, "val", f"{p}:{n}"), repo_id=task_repo_id, task_type="value", source="structural",
            question=rng.choice(TEMPLATES["value_const"]).format(name=n), source_id=f"const:{p}:{n}",
            grading=Grading(expected_paths=[p], expected_literal=lit, required_citations=[Span(path=p, start=line, end=line)])))
    used: set[str] = set()
    for f, p, lit in def_cands:
        if len(out["value"]) >= per_type:
            break
        if f.qual in used:
            continue
        used.add(f.qual)
        out["value"].append(Task(
            task_id=_tid(task_repo_id, "val", f"{f.sym.path}:{f.qual}:{p}"), repo_id=task_repo_id, task_type="value", source="structural",
            question=rng.choice(TEMPLATES["value_default"]).format(param=p, qual=f.qual, where=_where(facts, f.sym)),
            source_id=f"default:{f.sym.path}:{f.qual}:{p}",
            grading=Grading(expected_paths=[f.sym.path], expected_symbols=[f"{f.sym.path}:{f.qual}"], expected_literal=lit,
                            required_citations=[_span(f.sym)])))

    # enumerate: importers (2..8 files) and direct subclasses (2..8)
    imp_cands = [(m, v) for m, v in facts.importers.items() if 2 <= len({p for p, _ in v}) <= 8 and not _is_skipped(m)]
    sub_map: dict[str, list[IndexSymbol]] = defaultdict(list)
    class_by_name: dict[str, list[IndexSymbol]] = defaultdict(list)
    for s, _, _ in facts.classes:
        class_by_name[s.name].append(s)
    for s, _, bases in facts.classes:
        for b in bases:
            if len(class_by_name.get(b, [])) == 1:        # base defined exactly once in the repo
                sub_map[b].append(s)
    sub_cands = [(b, subs) for b, subs in sub_map.items() if 2 <= len(subs) <= 8 and not b.startswith("_")]
    rng.shuffle(imp_cands); rng.shuffle(sub_cands)
    stats["candidates"]["enumerate"] = len(imp_cands) + len(sub_cands)
    for m, v in imp_cands[:half]:
        files = sorted({p for p, _ in v})
        first_lines = {p: line for p, line in sorted(v, reverse=True)}
        out["enumerate"].append(Task(
            task_id=_tid(task_repo_id, "enum", f"importers:{m}"), repo_id=task_repo_id, task_type="enumerate", source="structural",
            question=rng.choice(TEMPLATES["enum_importers"]).format(dotted=facts.module_of_path.get(m, m)), source_id=f"importers:{m}",
            grading=Grading(expected_paths=files, required_citations=[Span(path=p, start=first_lines[p], end=first_lines[p]) for p in files[:5]])))
    for b, subs in sub_cands:
        if len(out["enumerate"]) >= per_type:
            break
        out["enumerate"].append(Task(
            task_id=_tid(task_repo_id, "enum", f"subclasses:{b}"), repo_id=task_repo_id, task_type="enumerate", source="structural",
            question=rng.choice(TEMPLATES["enum_subclasses"]).format(base=b), source_id=f"subclasses:{class_by_name[b][0].path}:{b}",
            grading=Grading(expected_paths=sorted({s.path for s in subs}), expected_symbols=sorted(f"{s.path}:{s.name}" for s in subs),
                            required_citations=[_span(s) for s in subs[:5]])))

    # trace: functions with 2..6 resolved repo-defined callees
    tr_cands = [f for f in facts.funcs if 2 <= len(f.callees) <= 6 and not f.sym.name.startswith("_")
                and f.sym.name not in GENERIC_NAMES and (f.sym.end - f.sym.start) <= 120]
    rng.shuffle(tr_cands)
    stats["candidates"]["trace"] = len(tr_cands)
    for f in tr_cands[:per_type]:
        out["trace"].append(Task(
            task_id=_tid(task_repo_id, "trace", f"{f.sym.path}:{f.qual}"), repo_id=task_repo_id, task_type="trace", source="structural",
            question=rng.choice(TEMPLATES["trace"]).format(qual=f.qual, where=_where(facts, f.sym)), source_id=f"callees:{f.sym.path}:{f.qual}",
            grading=Grading(expected_paths=sorted({c.split(":", 1)[0] for c in f.callees}), expected_symbols=sorted(f.callees),
                            required_citations=[_span(f.sym)])))

    # rebalance to the cap: fill shortfalls from types with surplus (locate keeps its 3x buffer until paraphrase)
    tasks: list[Task] = []
    for k in ("value", "enumerate", "trace"):
        tasks.extend(out[k][:per_type])
    tasks.extend(out["locate"])
    stats["generated"] = {k: len(v) for k, v in out.items()}
    return tasks, stats
