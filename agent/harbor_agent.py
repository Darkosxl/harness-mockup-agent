"""Frontier-bench agent — a thin subclass of Harbor's Terminus 2.

Terminus 2 is the reference terminal agent that ships inside Harbor: a tmux
session in the task container, a JSON tool-call protocol, context
summarization, and trajectory recording. Roughly 80KB of scaffolding that has
been tuned against Terminal-Bench. Rather than hand-roll a command loop, we
inherit all of it and only change the knobs.

The model comes from Harbor's `-m/--model` flag (a LiteLLM model string such as
`cerebras/gemma-4-31b`), so credentials stay in the runner's environment and
never in this repo.

Things worth tuning, in rough order of impact:
  max_turns                 how many commands the agent may run per task
  parser_name               "json" (default) or "xml" response protocol
  enable_summarize          keep long trajectories inside the context window
  temperature               0 is usually right for shell work
"""
from __future__ import annotations

from typing import Any

from harbor.agents.terminus_2 import Terminus2


class HarborAgent(Terminus2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # setdefault, never hard-code: the runner (or `--ak key=value`) must still win.
        kwargs.setdefault("max_turns", 40)
        kwargs.setdefault("temperature", 0.0)
        # asciinema recording needs extra binaries in every task image and buys us
        # nothing on a student leaderboard.
        kwargs.setdefault("record_terminal_session", False)
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "exposure-terminus-2"

    def version(self) -> str:
        return "3.0"
