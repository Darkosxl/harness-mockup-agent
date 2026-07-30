"""Standalone session of this agent — the harness runs this file twice.

  build stage   : once, must exit successfully inside 60s (structure + imports).
  RAM-bench     : 1 then 10 concurrent copies, peak PSS sampled over a fixed
                  window with HARNESS_RAM_PROBE=1 set. This is its own benchmark:
                  whatever main.py does here is what gets measured, so do a
                  representative slice of your agent's real work.

The two benchmark agents (agent/my_agent.py, agent/harbor_agent.py) import
packages that only exist inside their own environments (arcengine, harbor), so
they are structure-checked with ast instead of imported here.
"""
import ast
import os
import pathlib
import sys
import time

import requests

BASE = os.environ.get("HARNESS_LLM_BASE", "https://api.cerebras.ai/v1").rstrip("/")
KEY = os.environ.get("HARNESS_LLM_KEY", "")
MODEL = os.environ.get("HARNESS_LLM_MODEL", "gemma-4-31b")
RAM_PROBE = os.environ.get("HARNESS_RAM_PROBE") == "1"


def check_structure():
    for fname, cls in (("agent/my_agent.py", "MyAgent"),
                       ("agent/harbor_agent.py", "HarborAgent")):
        tree = ast.parse(pathlib.Path(fname).read_text())
        assert any(isinstance(n, ast.ClassDef) and n.name == cls for n in ast.walk(tree)), \
            f"{fname} must define class {cls}"
        print(f"ok: {fname} defines {cls}")


def think(prompt):
    """One reasoning step — the unit of work this agent's memory is measured on."""
    r = requests.post(
        f"{BASE}/chat/completions",
        headers={"Authorization": f"Bearer {KEY}"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": 64},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    check_structure()
    if not KEY:
        print("no HARNESS_LLM_KEY — structure check only")
        return
    # Keep reasoning for the whole probe window so the measurement is of a working
    # agent, not of a process that already exited. One shot outside the probe.
    deadline = time.time() + (25 if RAM_PROBE else 0)
    step = 0
    while True:
        step += 1
        try:
            answer = think(f"Step {step}: name one property of the number {step * 7}.")
            print(f"[step {step}] {answer[:80]}", file=sys.stderr)
        except Exception as e:
            print(f"[step {step}] error {e}", file=sys.stderr)
            time.sleep(2)
        if time.time() >= deadline:
            break
    print("session complete")


if __name__ == "__main__":
    main()
