"""Mockup agent for the Exposure Academy Agentic Harness.

Solves two small built-in puzzle sets by asking an OpenAI-compatible LLM
endpoint (Cerebras/SambaNova/anything). No tools, no memory — the simplest
thing that exercises the whole submission pipeline for real.

Env:
  HARNESS_LLM_BASE   default https://api.cerebras.ai/v1
  HARNESS_LLM_KEY    required
  HARNESS_LLM_MODEL  default gemma-4-31b-it
  HARNESS_SET        arc | frontier | all (default all)
  HARNESS_LIMIT      optional int, cap puzzles per set (RAM probe uses 2)

Prints one JSON object to stdout: {"score_arc": float|null, "score_frontier": float|null}
Progress goes to stderr.
"""

import json
import os
import sys

import requests

ARC_PUZZLES = [
    ("Continue the sequence: 2, 6, 18, 54. Answer with the next number only.", "162"),
    ("Reverse the string ABCD. Answer with the reversed string only.", "DCBA"),
    ("Continue the sequence: 1, 4, 9, 16, 25. Answer with the next number only.", "36"),
    ("If BLUE maps to EULB, what does GREEN map to? Answer with the mapped word only.", "NEERG"),
    ("Continue the sequence: 1, 1, 2, 3, 5, 8. Answer with the next number only.", "13"),
    ("Which is the odd one out: 3, 5, 9, 7, 11? Answer with the number only.", "9"),
]

FRONTIER_PUZZLES = [
    ("Which bash flag makes ls include hidden files? Answer with the flag only.", "-a"),
    ("What exit code does a successful unix command return? Answer with the number only.", "0"),
    ("What octal chmod value gives rwxr-xr-x? Answer with the number only.", "755"),
    ("Which git subcommand shows the working tree status? Answer with the subcommand only.", "status"),
    ("Which HTTP status code means Not Found? Answer with the number only.", "404"),
    ("In Python, what does len('hello') return? Answer with the number only.", "5"),
]


class MyAgent:
    def __init__(self):
        self.base = os.environ.get("HARNESS_LLM_BASE", "https://api.cerebras.ai/v1").rstrip("/")
        self.key = os.environ["HARNESS_LLM_KEY"]
        self.model = os.environ.get("HARNESS_LLM_MODEL", "gemma-4-31b-it")

    def ask(self, question):
        r = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": question}],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def solve_set(self, name, puzzles):
        limit = int(os.environ.get("HARNESS_LIMIT", "0")) or len(puzzles)
        puzzles = puzzles[:limit]
        correct = 0
        for i, (q, expected) in enumerate(puzzles, 1):
            try:
                answer = self.ask(q)
            except Exception as e:
                print(f"[{name} {i}/{len(puzzles)}] ERROR {e}", file=sys.stderr)
                continue
            ok = expected.lower() in answer.lower()
            correct += ok
            print(f"[{name} {i}/{len(puzzles)}] {'ok' if ok else 'MISS'} expected={expected!r} got={answer!r}",
                  file=sys.stderr)
        return round(100.0 * correct / len(puzzles), 1)

    def run(self):
        which = os.environ.get("HARNESS_SET", "all")
        result = {"score_arc": None, "score_frontier": None}
        if which in ("arc", "all"):
            result["score_arc"] = self.solve_set("arc", ARC_PUZZLES)
        if which in ("frontier", "all"):
            result["score_frontier"] = self.solve_set("frontier", FRONTIER_PUZZLES)
        print(json.dumps(result))
