"""B5 author phase: Sonnet explores a repo from a seed file with the SAME five tools (same caps) and writes one task.

Output of one attempt: question, reference answer, 3-5 atomic rubric items, task type, evidence spans. Evidence spans
must lie inside lines the teacher actually read (tools record them), so `required_citations` are real.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from tinker_cookbook.tool_use.types import ToolInput

from codeqa.agent.curation import DEFAULT_CAPS
from codeqa.agent.indexing.repomap import load_map
from codeqa.agent.prompts import user_prompt
from codeqa.agent.tools import RepoTools
from codeqa.clients.base import ModelClient
from codeqa.datagen import llm
from codeqa.shared.contracts import Message, Span

AUTHOR_TOOL_CALLS = 14
AUTHOR_TURNS = 10
CHAT_TIMEOUT = 90.0

AUTHOR_SYSTEM = """You write training questions about a code repository for a code Q&A assistant that answers with citations.
You have the same read-only tools the assistant will have. Explore starting from the seed file, read the relevant code
(line ranges, not whole files), follow one or two references if needed, then write ONE task.

A good task:
- is ONE question (not two joined by "and"), under 30 words, that a developer new to this repository would actually ask.
  Prefer these shapes, in this order: "Where is X handled / decided / validated?", "What happens when Y?",
  "What does Z do with W?", "Which component is responsible for ...?". Avoid "why" and "how does ... and what ...";
- cannot be answered from the question text alone: it needs the code you read;
- has a definite answer grounded in specific lines you read; prefer behaviour and control flow over trivia
  (no "what is the value of constant K", no docstring recitation);
- does not name the file path; it may name a public class or function if a developer would know it.

When you are done exploring, reply with NO tool call and a single JSON object:
{
  "task_type": "locate" | "trace" | "explain",
  "question": "...",
  "answer": "2-5 sentences; every factual claim followed by [path:Lstart-Lend] citing lines you read",
  "rubric": ["3-5 atomic yes/no facts a correct answer must state; each checkable independently; no compound items"],
  "evidence": [{"path": "...", "start": 41, "end": 67}]
}
`evidence` lists the 1-4 line ranges that establish the answer; they must be ranges you read with read_file."""


@dataclass
class AuthoredTask:
    task_type: str
    question: str
    answer: str
    rubric: list[str]
    evidence: list[Span]
    tool_calls: int = 0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    files_read: list[Span] = field(default_factory=list)


def _tool_text(result: Any) -> str:
    content = result.messages[0]["content"]
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def seed_prompt(seed_path: str, symbols: list[str]) -> str:
    syms = ", ".join(symbols[:12]) if symbols else "(no indexed symbols)"
    return (f"Seed file: {seed_path}\nSymbols defined there: {syms}\n\n"
            f"Start by reading the seed file (in line ranges). Then write the task as specified.")


def clip_to_read(evidence: list[Span], files_read: list[Span], max_lines: int = 80) -> list[Span]:
    """Keep the part of each evidence span the teacher actually read; drop spans it never read."""
    read_by_path: dict[str, set[int]] = {}
    for s in files_read:
        read_by_path.setdefault(s.path, set()).update(range(s.start, s.end + 1))
    out: list[Span] = []
    for e in evidence:
        lines = sorted(ln for ln in range(e.start, e.end + 1) if ln in read_by_path.get(e.path, set()))
        if not lines:
            continue
        start, end = lines[0], lines[-1]
        end = min(end, start + max_lines - 1)
        out.append(Span(path=e.path, start=start, end=end))
    return out


def parse_authored(text: str) -> dict[str, Any] | None:
    obj = llm.parse_json_object(text)
    if not obj:
        return None
    q = obj.get("question"); a = obj.get("answer"); rub = obj.get("rubric"); ev = obj.get("evidence")
    if not (isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip() and isinstance(rub, list) and isinstance(ev, list)):
        return None
    rub = [" ".join(str(r).split()) for r in rub if str(r).strip()]
    if not 3 <= len(rub) <= 6:
        return None
    spans: list[Span] = []
    for e in ev:
        try:
            spans.append(Span(path=str(e["path"]).lstrip("./"), start=int(e["start"]), end=int(e["end"])))
        except Exception:
            continue
    if not spans:
        return None
    tt = str(obj.get("task_type", "explain")).strip().lower()
    return {"task_type": tt if tt in ("locate", "trace", "explain") else "explain",
            "question": " ".join(q.split()), "answer": a.strip(), "rubric": rub[:5], "evidence": spans[:4]}


async def author(client: ModelClient, repo_id: str, seed_path: str, seed_symbols: list[str],
                 max_tool_calls: int = AUTHOR_TOOL_CALLS, max_turns: int = AUTHOR_TURNS) -> tuple[AuthoredTask | None, str]:
    """One authoring episode. Returns (task, reason) where reason is 'ok' or why it was rejected."""
    tools_obj = RepoTools(repo_id, max_tool_calls=max_tool_calls, caps=DEFAULT_CAPS)
    tools = {t.name: t for t in tools_obj.tools()}
    specs = tools_obj.specs()
    repo_map = load_map(repo_id)
    messages = [Message(role="system", content=AUTHOR_SYSTEM),
                Message(role="user", content=user_prompt(repo_id, repo_map, seed_prompt(seed_path, seed_symbols)))]
    ptoks = ctoks = 0
    final: Message | None = None
    for turn in range(1, max_turns + 1):
        msg = await asyncio.wait_for(client.chat(messages, tools=specs, max_tokens=2500), CHAT_TIMEOUT)
        ptoks += int(msg.usage.get("prompt_tokens", 0)); ctoks += int(msg.usage.get("completion_tokens", 0))
        messages.append(msg)
        if not msg.tool_calls:
            final = msg
            break
        remaining = max_tool_calls - tools_obj.calls
        calls = msg.tool_calls[: max(remaining, 0)]
        for i, tc in enumerate(calls):
            tc.call_id = tc.call_id or f"call_{turn}_{i}"
            tool = tools.get(tc.name)
            text = f"ERROR unknown_tool: {tc.name!r}" if tool is None else _tool_text(await tool.run(ToolInput(arguments=tc.args, call_id=tc.call_id)))
            messages.append(Message(role="tool", name=tc.name, content=text, call_id=tc.call_id))
        if len(calls) < len(msg.tool_calls) or remaining - len(calls) <= 0:
            messages.append(Message(role="user", content="No tool calls remain. Write the task JSON now."))
    else:
        return None, "max_turns"
    if final is None:
        # budget exhausted: one last turn without tools
        msg = await asyncio.wait_for(client.chat(messages, tools=None, max_tokens=2500), CHAT_TIMEOUT)
        ptoks += int(msg.usage.get("prompt_tokens", 0)); ctoks += int(msg.usage.get("completion_tokens", 0))
        final = msg
    parsed = parse_authored(final.content)
    if parsed is None:                                    # one retry: ask for the bare JSON object
        messages.append(Message(role="user", content="Reply with only the JSON object described in the instructions, nothing else."))
        msg = await asyncio.wait_for(client.chat(messages, tools=None, max_tokens=2500), CHAT_TIMEOUT)
        ptoks += int(msg.usage.get("prompt_tokens", 0)); ctoks += int(msg.usage.get("completion_tokens", 0))
        messages.append(msg)
        parsed = parse_authored(msg.content)
    if parsed is None:
        return None, "unparseable"
    evidence = clip_to_read(parsed["evidence"], tools_obj.files_read)
    if not evidence:
        return None, "evidence_not_read"
    if re.search(r"\.py\b", parsed["question"]):
        return None, "question_names_path"
    return AuthoredTask(task_type=parsed["task_type"], question=parsed["question"], answer=parsed["answer"], rubric=parsed["rubric"],
                        evidence=evidence, tool_calls=tools_obj.calls, turns=len([m for m in messages if m.role == "assistant"]),
                        prompt_tokens=ptoks, completion_tokens=ctoks, files_read=list(tools_obj.files_read)), "ok"
