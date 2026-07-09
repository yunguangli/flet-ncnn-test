# Task Completion Verification

After any code change, run:

1. `uv run flet run --web` — launch and manually test the app (click Start Detection, verify UI updates and no traceback in server log)
2. `uv run pytest` — if pytest tests exist in `tests/`

No linter or formatter is configured. No type checker beyond what Flet's runtime enforces.

Do **not** commit changes unless explicitly asked.
