"""Frontier-bench (Harbor) agent for the Exposure Academy Agentic Harness.

Runs on the host inside Harbor's own Python — standard library ONLY
(requirements.txt is NOT installed for this file). Solves a task by asking an
OpenAI-compatible LLM for shell commands and executing them in the task
container via environment.exec(), Terminus-style, up to MAX_STEPS.

Env (injected by the harness via --ae): HARNESS_LLM_BASE / KEY / MODEL.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from harbor.agents.base import BaseAgent

MAX_STEPS = 20
SYSTEM = (
    "You are an autonomous agent solving a task in a Linux container. "
    "Reply ONLY with a JSON object: {\"cmd\": \"<shell command>\"} to run a "
    "command, or {\"done\": true} when the task is complete. One command at a "
    "time; non-interactive commands only."
)


class HarborAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "exposure-mockup"

    def version(self) -> str:
        return "2.0"

    def _env(self, key: str, default: str = "") -> str:
        return self.extra_env.get(key) or os.environ.get(key, default)

    def _llm(self, messages: list[dict]) -> str:
        base = self._env("HARNESS_LLM_BASE", "https://api.cerebras.ai/v1").rstrip("/")
        body = json.dumps({
            "model": self._env("HARNESS_LLM_MODEL", "gemma-4-31b"),
            "messages": messages,
            "temperature": 0,
            "max_tokens": 300,
        }).encode()
        req = urllib.request.Request(
            base + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + self._env("HARNESS_LLM_KEY"),
                     "Content-Type": "application/json",
                     # urllib's default User-Agent gets 403'd by the API's CDN
                     "User-Agent": "harness-agent/1.0"})
        for attempt in range(5):  # 429/5xx backoff — rate limits must not kill the trial
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r)["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        return ""

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Task:\n" + instruction}]
        steps = 0
        for _ in range(MAX_STEPS):
            reply = self._llm(messages)
            messages.append({"role": "assistant", "content": reply})
            try:
                start = reply.index("{")
                cmd = json.loads(reply[start:reply.rindex("}") + 1])
            except ValueError:
                messages.append({"role": "user", "content": "Reply with JSON only."})
                continue
            if cmd.get("done"):
                break
            if not cmd.get("cmd"):
                messages.append({"role": "user", "content": "Missing 'cmd'."})
                continue
            steps += 1
            try:
                res = await environment.exec(cmd["cmd"], timeout_sec=180)
                out = f"exit={res.return_code}\nstdout:\n{(res.stdout or '')[-2000:]}\nstderr:\n{(res.stderr or '')[-1000:]}"
            except Exception as e:
                out = f"exec error: {e}"
            messages.append({"role": "user", "content": out})
        context.metadata = {"steps": steps}
