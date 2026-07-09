# Conventions

## Style
- Prefer no new inline comments; existing modules may have docstrings (keep consistent when editing a file)
- Type hints everywhere; `X | None` over Optional
- dataclasses for data; `@property` for computed fields; `@staticmethod` for pure helpers
- `async def` for Flet handlers and I/O-bound loops

## Naming
- snake_case modules/functions; PascalCase classes; UPPER_SNAKE constants
- Private methods: single `_` prefix

## Patterns
- CVM: Model → Controller → View
- Controller → View callbacks via `on_*` setters; View wires in `_wire_callbacks()`
- Controller owns state; View refreshes with `page.update()` after mutations
- `logger = logging.getLogger(__name__)` per module
- Errors: Controller `_emit_error()` → View status text
- Single pyproject.toml; no extra requirements files
