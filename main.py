"""Smoke test — the harness runs this during the `building` stage (60s budget).

The two agents import framework packages (arcengine / harbor) that only exist
inside their benchmark environments, so this checks structure with `ast`
instead of importing them: both files parse and define the required classes.
"""
import ast
import pathlib

for fname, cls in (("agent/my_agent.py", "MyAgent"),
                   ("agent/harbor_agent.py", "HarborAgent")):
    tree = ast.parse(pathlib.Path(fname).read_text())
    assert any(isinstance(n, ast.ClassDef) and n.name == cls for n in ast.walk(tree)), \
        f"{fname} must define class {cls}"
    print(f"ok: {fname} defines {cls}")

print("smoke test passed")
