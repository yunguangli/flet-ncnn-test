# Task Completion Verification

After code changes:

1. `uv run flet run --web` — manual smoke: Start Camera, Start Detection; no traceback in terminal
2. `uv run pytest` — if `tests/` still applies to changed surface (current `tests/test_main.py` is a counter stub, not app logic)

No project linter/formatter/typechecker configured.

Do not commit unless the user asks.
