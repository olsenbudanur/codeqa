"""Prove each external client works with the keys in .env. Run: uv run python -m scripts.smoke_clients

Each check is independent; a missing key is reported, not fatal.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

from codeqa.shared.contracts import EndpointProfile, Message

load_dotenv()
results: dict[str, str] = {}


def report(name: str, ok: bool, detail: str):
    results[name] = ("OK   " if ok else "FAIL ") + detail
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")


async def check_anthropic():
    print("\n=== Anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return report("anthropic", False, "ANTHROPIC_API_KEY missing")
    from codeqa.clients.anthropic import AnthropicClient
    for model in ("claude-haiku-4-5-20251001", "claude-sonnet-5"):
        try:
            c = AnthropicClient(EndpointProfile(name="t", kind="anthropic", model=model, max_generation_tokens=32, thinking=False))
            t0 = time.time()
            m = await c.chat([Message(role="user", content="Reply with the single word: ready")], temperature=0)
            report(f"anthropic:{model}", "ready" in m.content.lower(), f"{m.content.strip()[:40]!r} in {time.time()-t0:.1f}s")
        except Exception as e:
            report(f"anthropic:{model}", False, f"{type(e).__name__}: {str(e)[:120]}")
    # tool use round trip
    try:
        c = AnthropicClient(EndpointProfile(name="t", kind="anthropic", model="claude-haiku-4-5-20251001", max_generation_tokens=128, thinking=False))
        tools = [{"name": "find_symbol", "description": "Find a symbol definition", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}]
        m = await c.chat([Message(role="user", content="Use the find_symbol tool to look up 'validate_session'.")], tools=tools, temperature=0)
        report("anthropic:tool_call", bool(m.tool_calls) and m.tool_calls[0].name == "find_symbol", f"tool_calls={[(t.name, t.args) for t in m.tool_calls]}")
    except Exception as e:
        report("anthropic:tool_call", False, f"{type(e).__name__}: {str(e)[:120]}")


async def _stage(name: str, coro_or_fn, seconds: int):
    """Run one stage with a hard timeout; print immediately either way."""
    t0 = time.time()
    try:
        if asyncio.iscoroutine(coro_or_fn):
            out = await asyncio.wait_for(coro_or_fn, timeout=seconds)
        else:
            out = await asyncio.wait_for(asyncio.to_thread(coro_or_fn), timeout=seconds)
        return out, time.time() - t0
    except asyncio.TimeoutError:
        report(name, False, f"timeout after {seconds}s")
        return None, seconds
    except Exception as e:
        report(name, False, f"{type(e).__name__}: {str(e)[:160]}")
        return None, time.time() - t0


async def check_tinker():
    print("\n=== Tinker (staged, each with a hard timeout)")
    if not os.environ.get("TINKER_API_KEY"):
        return report("tinker", False, "TINKER_API_KEY missing")
    from codeqa.clients import tinker as tk
    caps, dt = await _stage("tinker:capabilities", lambda: tk.service_client().get_server_capabilities(), 20)
    if caps is None:
        return
    models = [getattr(m, "model_name", str(m)) for m in getattr(caps, "supported_models", [])]
    has4b = any("Qwen3.5-4B" in m for m in models)
    report("tinker:capabilities", has4b, f"{len(models)} models in {dt:.1f}s; Qwen3.5-4B {'present' if has4b else 'MISSING'}")
    r, dt = await _stage("tinker:renderer", lambda: tk.renderer("Qwen/Qwen3.5-4B"), 90)
    if r is None:
        return
    report("tinker:renderer", True, f"{type(r).__name__} + tokenizer in {dt:.1f}s (first run downloads the tokenizer)")
    text, dt = await _stage("tinker:sample", tk.sample_text("Qwen/Qwen3.5-4B", [{"role": "user", "content": "Say 'ready' and nothing else."}], max_tokens=32, temperature=0.0), 90)
    if text is None:
        return
    report("tinker:sample", len(text) > 0, f"{text.strip()[:80]!r} in {dt:.1f}s")


def check_modal():
    print("\n=== Modal")
    try:
        out = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True, timeout=60)
        ok = out.returncode == 0 and out.stdout.strip() != ""
        report("modal:profile", ok, (out.stdout or out.stderr).strip()[:120] or "no profile; run `uv run modal setup`")
    except Exception as e:
        report("modal:profile", False, f"{type(e).__name__}: {str(e)[:120]}")
    try:
        out = subprocess.run([sys.executable, "-m", "modal", "secret", "list"], capture_output=True, text=True, timeout=60)
        report("modal:secrets", out.returncode == 0, ("has 'codeqa'" if "codeqa" in out.stdout else "secret 'codeqa' not found") if out.returncode == 0 else out.stderr.strip()[:120])
    except Exception as e:
        report("modal:secrets", False, f"{type(e).__name__}: {str(e)[:120]}")


def check_github_hf():
    print("\n=== GitHub / Hugging Face")
    from codeqa.clients import github, hf
    try:
        sha = github.resolve_sha("pallets", "flask", "main")
        report("github", len(sha) == 40, f"token={'yes' if github.token() else 'no'}; flask main={sha[:7]}")
    except Exception as e:
        report("github", False, f"{type(e).__name__}: {str(e)[:120]}")
    try:
        n = hf.size("Qodo/deep_code_bench", "default", "train")
        report("hf", n == 912, f"deep_code_bench train rows={n}; HF_TOKEN={'yes' if os.environ.get('HF_TOKEN') else 'no (fine, public)'}")
    except Exception as e:
        report("hf", False, f"{type(e).__name__}: {str(e)[:120]}")


async def timed(name: str, coro, seconds: int = 120):
    try:
        await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        report(f"{name}:timeout", False, f"no response within {seconds}s")


async def main():
    check_github_hf()
    await timed("anthropic", check_anthropic(), 60)
    await check_tinker()
    check_modal()
    print("\n=== Summary")
    for k, v in results.items():
        print(f"  {v:60s} {k}" if False else f"  {k:24s} {v}")
    print("\nWandB:", "WANDB_API_KEY set" if os.environ.get("WANDB_API_KEY") else "WANDB_API_KEY missing (optional)")


if __name__ == "__main__":
    asyncio.run(main())
