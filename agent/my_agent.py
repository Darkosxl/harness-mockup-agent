"""ARC-AGI-3 agent — a thin subclass of the framework's own LLM agent.

`agents.templates.llm_agents.LLM` ships inside ARC-AGI-3-Agents (vendored by the
starter kit). It already handles the whole loop: chat history with a rolling
window, the RESET bootstrap, one OpenAI tool per game action, coordinate parsing
for ACTION6, and token accounting. We inherit it and only supply strategy.

The LLM endpoint comes from OPENAI_BASE_URL / OPENAI_API_KEY, which the runner
injects — the OpenAI SDK reads both from the environment, so any
OpenAI-compatible provider works and no key ever lives in this repo.

Things worth tuning, in rough order of impact:
  build_user_prompt   what the model is told about the game — the real lever
  DO_OBSERVATION      True = a reasoning call before each action (2x the calls)
  MESSAGE_LIMIT       how much history the model still sees
  MAX_ACTIONS         per-game action budget (the runner caps this too)
"""
from __future__ import annotations

import os
import textwrap

from arcengine import FrameData

from agents.templates.llm_agents import LLM


class MyAgent(LLM):
    MAX_ACTIONS = 200
    MODEL = os.environ.get("HARNESS_LLM_MODEL", "gemma-4-31b")
    # gemma answers with tool_calls, not the legacy function_call field.
    MODEL_REQUIRES_TOOLS = True
    # One call per action instead of two: an observation pass doubles both the
    # wall clock and the rate-limit pressure across 25 games. Flip it back on if
    # your model reasons better when it narrates first.
    DO_OBSERVATION = False
    MESSAGE_LIMIT = 12

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            f"""
# CONTEXT:
You are playing an unfamiliar 2D puzzle game. You must discover the rules by
experimenting. Your objective is to complete levels (raise levels_completed)
and reach WIN, using as few actions as possible, without hitting GAME_OVER.

The screen is a grid of integers. Each cell is INT<0,15>: the value is a colour
id, and shapes of equal value are the same object. Coordinates are
INT<0,63> for both x and y, with (0,0) at the top-left.

# WHAT YOU KNOW SO FAR:
state={latest_frame.state.name} levels_completed={latest_frame.levels_completed}
available actions this turn: {latest_frame.available_actions}

# STRATEGY:
- Early on, prefer unused simple actions to learn what each one changes.
- Compare the last grid to the current one: the cells that changed reveal
  which object you control and what the action did.
- If an action changed nothing twice, stop repeating it and try another.
- ACTION6 clicks a specific cell — use it to interact with an object you have
  already located, not to explore blindly.
- If the state is GAME_OVER, call RESET and avoid whatever preceded the loss.

# TURN:
Call exactly one action.
        """
        ).strip()
