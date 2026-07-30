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
  pretty_print_3d     how a frame is encoded; this dominates token cost
  DO_OBSERVATION      True = a reasoning call before each action (2x the calls)
  MESSAGE_LIMIT       how much history the model still sees
  MAX_ACTIONS         per-game action budget (the runner caps this too)
"""
from __future__ import annotations

import logging
import os
import random
import textwrap
import time
from typing import Any

import openai
from arcengine import FrameData, GameAction

from agents.templates.llm_agents import LLM

logger = logging.getLogger(__name__)

HEX = "0123456789abcdef"


class MyAgent(LLM):
    MAX_ACTIONS = 200
    MODEL = os.environ.get("HARNESS_LLM_MODEL", "gemma-4-31b")
    # gemma answers with tool_calls, not the legacy function_call field.
    MODEL_REQUIRES_TOOLS = True
    # One call per action instead of two: an observation pass doubles both the
    # wall clock and the rate-limit pressure across 25 games. Flip it back on if
    # your model reasons better when it narrates first.
    DO_OBSERVATION = False
    MESSAGE_LIMIT = 6

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        """One hex char per cell instead of a Python list repr.

        The framework prints every row as `[0, 0, 10, ...]`, which costs roughly
        10k tokens for a 64x64 grid — the brackets, commas and spaces outweigh the
        data. Cell values are INT<0,15>, so a single hex digit is lossless and cuts
        a frame to about 1.5k tokens. Only the final grid is sent: it is the
        current state, and history already lives in the message window.
        """
        if not array_3d:
            return "(empty)"
        rows = ["".join(HEX[int(v) & 15] for v in row) for row in array_3d[-1]]
        return ("Grid, one hex char per cell, row 0 first, column 0 leftmost:\n"
                + "\n".join(rows))

    # The template builds its own OpenAI client per call with max_retries=2, so a
    # provider that is merely busy (429 queue_exceeded) ends the whole game. A run is
    # thousands of calls long — absorb throttling here or lose hours of play to one
    # bad minute. Acting blind beats dying: a random legal action keeps the game alive.
    RATE_LIMIT_TRIES = 6

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        for attempt in range(self.RATE_LIMIT_TRIES):
            try:
                return super().choose_action(frames, latest_frame)
            except (openai.RateLimitError, openai.APIConnectionError,
                    openai.InternalServerError) as e:
                wait = min(2 ** attempt, 30)
                logger.warning(f"provider unavailable ({type(e).__name__}), "
                               f"retry {attempt + 1}/{self.RATE_LIMIT_TRIES} in {wait}s")
                time.sleep(wait)
            except openai.BadRequestError as e:
                # FIFO trimming can cut between an assistant tool_calls message and its
                # tool response, which the API rejects outright. The template's guard
                # only strips leading tool messages and doesn't catch every case, so
                # drop the history and let the next call rebuild it — losing context
                # beats losing the remaining games.
                logger.warning(f"history rejected ({str(e)[:120]}) — resetting messages")
                self.messages = []
        legal = latest_frame.available_actions or [1, 2, 3, 4, 5, 6]
        action = GameAction.from_name(f"ACTION{random.choice(legal)}")
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        action.reasoning = {"fallback": "provider throttled, acting blind"}
        return action

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
