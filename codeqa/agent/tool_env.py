"""v3 training env: the cookbook's AgentToolMessageEnv plus rounds, a context cap, the remaining-budget trailer and the
forced final answer (rounds.py holds the pure mechanics; driver.py applies the same ones on the product path).

Round = one assistant message with up to `commands_per_turn` tool calls (all run, all count as one round). After the
round's tool results the env appends "[context 14k of 32k; 9 message(s) left]". When the next round would blow the
context cap, or exactly one message is left, it appends FORCED_PROMPT as a user message; the next assistant message is
the final answer, and one that still calls tools ends the episode with no answer (stop max_turns, graded as such).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tinker_cookbook.renderers.base import Message, get_text_content
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_presets import default_rollout_config_for_model
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

from codeqa.agent import rounds


def _append_text(msg: Message, text: str) -> None:
    content = msg.get("content", "")
    if isinstance(content, list):
        content.append({"type": "text", "text": text})
    else:
        msg["content"] = (content or "") + text


def _set_text(msg: Message, text: str) -> None:
    msg["content"] = text


@dataclass
class CodeQAToolMessageEnv(AgentToolMessageEnv):
    renderer: Any = None
    context_cap: int | None = None          # prompt tokens of the next turn; None = classic call-budget behaviour
    commands_per_turn: int | None = None
    output_cap_chars: int = 12000
    _forced: bool = field(default=False, init=False)

    def context_tokens(self) -> int:
        try:
            return int(self.renderer.build_generation_prompt(self.history).length)
        except Exception:  # noqa: BLE001  (renderer unavailable in tests)
            return sum(rounds.approx_tokens(get_text_content(m)) for m in self.history)

    async def step(self, message: Message) -> MessageStepResult:
        if self.context_cap is None:
            return await super().step(message)
        tool_calls = list(message.get("tool_calls") or [])
        if self._forced and tool_calls:                      # the forced turn still called tools: no answer, episode over
            self._turn_count += 1
            self.history.append(message)
            metrics = {"max_turns": 1.0, "forced_no_answer": 1.0, f"{types.STOP_METRIC_PREFIX}{types.StopReason.MAX_TURNS}": 1.0}
            reward, reward_metrics = await self._grade()
            metrics.update(reward_metrics)
            return MessageStepResult(reward=reward, episode_done=True, next_messages=self.history, metrics=metrics, logs={})
        kept, dropped = rounds.cap_calls(tool_calls, self.commands_per_turn)
        to_step = message
        if dropped:
            to_step = dict(message)
            to_step["tool_calls"] = kept
        idx = len(self.history)
        result = await super().step(to_step)
        if dropped:
            self.history[idx] = message                      # keep what the model actually said; only `kept` ran
        tool_msgs = [m for m in self.history[idx + 1:] if m.get("role") == "tool"]
        if tool_msgs:
            texts = rounds.cap_outputs([get_text_content(m) for m in tool_msgs], self.output_cap_chars)
            for m, t in zip(tool_msgs, texts):
                _set_text(m, t)
            if dropped:
                _append_text(tool_msgs[-1], rounds.COMMAND_CAP_NOTE.format(dropped=dropped, cap=self.commands_per_turn))
        if result.episode_done:
            return result
        tokens = self.context_tokens()
        left = self.max_turns - self._turn_count
        if tool_msgs:
            _append_text(tool_msgs[-1], rounds.trailer(tokens, self.context_cap, left))
        metrics = dict(result.metrics)
        metrics["context_tokens"] = float(tokens)
        if rounds.should_force(tokens, self.context_cap, self._turn_count, self.max_turns):
            self.history.append({"role": "user", "content": rounds.FORCED_PROMPT})
            self._forced = True
            metrics["forced_answer_prompted"] = 1.0
        return MessageStepResult(reward=result.reward, episode_done=False, next_messages=self.history, metrics=metrics, logs=result.logs)


def build_codeqa_tool_env(*, renderer, tools, initial_messages, reward_fn, max_turns: int, max_tool_calls: int | None,
                          max_generation_tokens: int | None, max_trajectory_tokens: int | None, model_name: str | None,
                          context_cap: int, commands_per_turn: int | None, output_cap_chars: int = 12000,
                          failed_parse_reward: float = -0.1, terminate_on_parse_error: bool = True,
                          context_overflow_reward: float = -0.1) -> EnvFromMessageEnv:
    """Mirror of tinker_cookbook's build_agent_tool_env with CodeQAToolMessageEnv inside (same rollout-config resolution)."""
    cfg = default_rollout_config_for_model(model_name) if model_name is not None else None
    parse_error_policy = cfg.parse_errors if cfg is not None else None
    if cfg is not None and cfg.limits.max_tool_calls is not None:
        max_tool_calls = cfg.limits.max_tool_calls if max_tool_calls is None else min(max_tool_calls, cfg.limits.max_tool_calls)
    runner_owns_length = cfg is not None and (cfg.limits.max_trajectory_tokens is not None or cfg.limits.max_sampled_tokens is not None
                                              or cfg.limits.max_turn_tokens is not None)
    msg_env = CodeQAToolMessageEnv(
        tools=tools, initial_messages=initial_messages, max_turns=max_turns, reward_fn=reward_fn,
        failed_parse_reward=failed_parse_reward, terminate_on_parse_error=terminate_on_parse_error, max_tool_calls=max_tool_calls,
        parse_error_policy=parse_error_policy, tool_execution=cfg.tool_execution if cfg is not None else "parallel",
        termination_policy=cfg.termination if cfg is not None else None,
        renderer=renderer, context_cap=context_cap, commands_per_turn=commands_per_turn, output_cap_chars=output_cap_chars,
    )
    return EnvFromMessageEnv(
        renderer=renderer, message_env=msg_env, failed_parse_reward=failed_parse_reward, terminate_on_parse_error=terminate_on_parse_error,
        max_trajectory_tokens=max_trajectory_tokens, max_generation_tokens=max_generation_tokens,
        context_overflow_reward=context_overflow_reward, terminate_on_length=not runner_owns_length,
        parse_error_policy=parse_error_policy, rollout_limits=cfg.limits if cfg is not None else None,
    )
