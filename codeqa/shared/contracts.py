"""Every schema in the project. Components import from here and nowhere else.

Contract numbers (C1..C10) match docs/contracts.md.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+__[A-Za-z0-9_.\-]+__[0-9a-f]{7}(__nodoc)?$")

# C4: citations look like [path:L10-L20] or [path:L10]
CITATION_RE = re.compile(r"\[([^\]\s:]+):L(\d+)(?:-L(\d+))?\]")


def make_repo_id(owner: str, repo: str, sha: str, nodoc: bool = False) -> str:
    rid = f"{owner}__{repo}__{sha[:7]}"
    return rid + "__nodoc" if nodoc else rid


class Span(BaseModel):
    path: str
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: int, info) -> int:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError("end must be >= start")
        return v


# ---------------------------------------------------------------------------
# C1: snapshot manifest
# ---------------------------------------------------------------------------

class FileEntry(BaseModel):
    path: str
    lang: str | None = None
    lines: int
    bytes: int


class Manifest(BaseModel):
    repo_id: str
    url: str
    sha: str
    files: list[FileEntry]
    dropped: dict[str, int] = Field(default_factory=dict)

    @field_validator("repo_id")
    @classmethod
    def _rid(cls, v: str) -> str:
        if not REPO_ID_RE.match(v):
            raise ValueError(f"bad repo_id {v!r}; expected <owner>__<repo>__<sha7>")
        return v


# ---------------------------------------------------------------------------
# C2: index
# ---------------------------------------------------------------------------

SymbolKind = Literal["function", "class", "method"]


class IndexSymbol(BaseModel):
    name: str
    kind: SymbolKind
    path: str
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    parent: str | None = None
    signature: str = ""

    @property
    def qualified(self) -> str:
        """CodeScout-style 'path:Class.method' or 'path:function'."""
        return f"{self.path}:{self.parent + '.' if self.parent else ''}{self.name}"


# ---------------------------------------------------------------------------
# C5: task record
# ---------------------------------------------------------------------------

TaskType = Literal["locate", "value", "enumerate", "trace", "explain"]
TaskSource = Literal["deepcodebench", "codescout", "structural", "teacher", "sweqa", "sweqa_pro"]
Split = Literal["train", "eval"]


class Grading(BaseModel):
    expected_paths: list[str] = Field(default_factory=list)      # verifiable: any-of or F1 when >1
    expected_symbols: list[str] = Field(default_factory=list)    # 'path:Class.method'; F1 when >1
    expected_literal: str | None = None                          # value type; normalized string match
    reference_answer: str | None = None                          # judged types
    rubric: list[str] = Field(default_factory=list)              # atomic yes/no items; DeepCodeBench facts go here
    required_citations: list[Span] = Field(default_factory=list)

    @property
    def is_verifiable(self) -> bool:
        return bool(self.expected_paths or self.expected_symbols or self.expected_literal is not None)


class Budget(BaseModel):
    max_tool_calls: int = 10
    max_turns: int = 8
    max_answer_tokens: int = 400


# Answer caps raised 2026-09-18 evening (decisions.md): at 450 Claude's SWE-QA explain answers failed the format gate
# on length alone (p50 671 tokens); at 800 it passes 71 % / DeepCodeBench 98 %. Base Qwen answers average ~190 tokens,
# so the caps only bind on the teacher and the strong baseline. The cap is in the prompt: change it only between runs.
DEFAULT_BUDGETS: dict[str, Budget] = {
    "locate": Budget(max_tool_calls=6, max_turns=6, max_answer_tokens=400),
    "value": Budget(max_tool_calls=6, max_turns=6, max_answer_tokens=300),
    "enumerate": Budget(max_tool_calls=10, max_turns=8, max_answer_tokens=500),
    "trace": Budget(max_tool_calls=10, max_turns=8, max_answer_tokens=600),
    "explain": Budget(max_tool_calls=12, max_turns=10, max_answer_tokens=800),
}


class Task(BaseModel):
    task_id: str
    repo_id: str
    split: Split = "train"
    question: str
    task_type: TaskType
    source: TaskSource
    source_id: str | None = None
    grading: Grading = Field(default_factory=Grading)
    budget: Budget | None = None

    @field_validator("repo_id")
    @classmethod
    def _rid(cls, v: str) -> str:
        if not REPO_ID_RE.match(v):
            raise ValueError(f"bad repo_id {v!r}")
        return v

    def effective_budget(self) -> Budget:
        return self.budget or DEFAULT_BUDGETS[self.task_type]


# ---------------------------------------------------------------------------
# C6: episode trace
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["answer", "max_turns", "budget", "overflow", "parse_error", "error"]


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class Message(BaseModel):
    role: Role
    content: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    name: str | None = None          # tool name for role == "tool"
    call_id: str | None = None       # matches ToolCall.call_id for role == "tool"
    parse_error: str | None = None   # assistant only: malformed tool call or truncated generation (raw text kept)
    usage: dict[str, int] = Field(default_factory=dict)  # assistant only: prompt_tokens, completion_tokens


class TraceStats(BaseModel):
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    files_read: list[Span] = Field(default_factory=list)
    stop_reason: StopReason = "answer"
    seconds: float = 0.0


class Trace(BaseModel):
    task_id: str
    profile: str
    messages: list[Message]
    stats: TraceStats = Field(default_factory=TraceStats)
    answer: str = ""


# ---------------------------------------------------------------------------
# C7: grade result
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    path: str
    start: int
    end: int
    exists: bool = False
    grounded: bool = False      # the cited range was read in this episode
    anchors_symbol: bool = False  # contains a gold symbol definition (identifier grounding)


class CitationReport(BaseModel):
    citations: list[Citation] = Field(default_factory=list)

    @property
    def any_(self) -> bool:
        return bool(self.citations)

    @property
    def all_exist(self) -> bool:
        return bool(self.citations) and all(c.exists for c in self.citations)

    @property
    def all_grounded(self) -> bool:
        return bool(self.citations) and all(c.grounded for c in self.citations)


class GradeComponents(BaseModel):
    format_ok: float = 0.0
    citations_parse: float = 0.0
    citations_exist: float = 0.0
    citations_grounded: float = 0.0
    identifier_grounded: float = 0.0
    correctness: float = 0.0
    efficiency: float = 1.0


GateName = Literal["format", "citations", "grounding", "budget", "judge_error"]


class GradeResult(BaseModel):
    reward: float                      # NaN allowed: judge failure; trainer maps NaN -> group mean
    components: GradeComponents = Field(default_factory=GradeComponents)
    gate_failed: GateName | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# C8: endpoint profile
# ---------------------------------------------------------------------------

ClientKind = Literal["openai", "anthropic", "tinker"]


class EndpointProfile(BaseModel):
    name: str
    kind: ClientKind
    model: str                          # served name, base model name, or tinker:// checkpoint path
    base_model: str | None = None       # tinker kind with a tinker:// path: the base model for tokenizer + renderer
    base_url: str | None = None
    api_key_env: str | None = None      # env var holding the key
    renderer: str | None = None         # cookbook renderer name (tinker kind)
    tool_parser: str | None = None      # vLLM flag (openai kind)
    max_context: int = 65536
    max_generation_tokens: int = 2048
    thinking: bool = True


# ---------------------------------------------------------------------------
# C9: product stream events
# ---------------------------------------------------------------------------

EventType = Literal["thinking", "tool_call", "tool_result", "answer", "citations", "stats", "done", "error"]


class SSEEvent(BaseModel):
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Checkpoint manifest (gap_specs §2)
# ---------------------------------------------------------------------------

class CheckpointRecord(BaseModel):
    name: str
    run: str
    step: int
    created_at: datetime
    tinker_path: str
    merged_path: str | None = None
    modal_volume: str | None = None
    profile: str | None = None
    evals: dict[str, dict[str, float]] = Field(default_factory=dict)
    is_final: bool = False
    notes: str = ""
