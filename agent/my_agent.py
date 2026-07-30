"""ARC-AGI-3 agent for the Exposure Academy Agentic Harness (reference example).

Runs inside the ARC-AGI-3-Agents framework (the harness overlays this file onto
the official Kaggle starter and drives it with scripts/play_local.py). Encodes
the current frame as a small text grid, asks an OpenAI-compatible LLM for the
next action, falls back to a random legal action on any error.

Env (injected by the harness):
  HARNESS_LLM_BASE / HARNESS_LLM_KEY / HARNESS_LLM_MODEL
"""
from __future__ import annotations

import os
import random
import re
from typing import Any

import requests

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

SYSTEM = (
    "You control an agent in a 2D puzzle game shown as a grid of digits. "
    "Reply with exactly one action token: ACTION1, ACTION2, ACTION3, ACTION4, "
    "ACTION5, or ACTION6 x y (coordinates 0-63). Nothing else."
)


def encode_grid(frame: list[list[list[int]]]) -> str:
    """Downsample the 64x64 top layer to 16x16 so the prompt stays tiny."""
    if not frame:
        return "(empty)"
    g = frame[-1]
    rows = []
    for y in range(0, len(g), 4):
        rows.append("".join(str(g[y][x] % 10) for x in range(0, len(g[y]), 4)))
    return "\n".join(rows)


class MyAgent(Agent):
    """Asks the LLM each step; random fallback keeps the run alive on errors."""

    MAX_ACTIONS = 40

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.base = os.environ.get("HARNESS_LLM_BASE", "https://api.cerebras.ai/v1").rstrip("/")
        self.key = os.environ.get("HARNESS_LLM_KEY", "")
        self.model = os.environ.get("HARNESS_LLM_MODEL", "gemma-4-31b")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def ask_llm(self, latest_frame: FrameData) -> str:
        legal = latest_frame.available_actions or [1, 2, 3, 4, 5, 6]
        prompt = (
            f"Legal actions: {', '.join('ACTION%d' % a for a in legal)}\n"
            f"Levels completed: {latest_frame.levels_completed}\n"
            f"Grid (16x16 downsample):\n{encode_grid(latest_frame.frame)}\n"
            "Next action?"
        )
        r = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 16,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        legal = latest_frame.available_actions or [1, 2, 3, 4, 5, 6]
        try:
            reply = self.ask_llm(latest_frame)
            m = re.search(r"ACTION([1-6])(?:\D+(\d{1,2})\D+(\d{1,2}))?", reply)
            n = int(m.group(1)) if m and int(m.group(1)) in legal else random.choice(legal)
            action = GameAction.from_name(f"ACTION{n}")
            if action.is_complex():
                x = min(int(m.group(2)), 63) if m and m.group(2) else random.randint(0, 63)
                y = min(int(m.group(3)), 63) if m and m.group(3) else random.randint(0, 63)
                action.set_data({"x": x, "y": y})
            action.reasoning = {"llm": reply[:120]}
        except Exception as e:  # LLM down/rate-limited → keep playing randomly
            n = random.choice(legal)
            action = GameAction.from_name(f"ACTION{n}")
            if action.is_complex():
                action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"fallback": str(e)[:120]}
        return action
