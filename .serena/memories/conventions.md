# Conventions

## Code style
- **No** inline comments (`#` annotations permitted for section dividers only)
- Type hints everywhere, prefer Python 3.10+ union syntax (`X | None`)
- `dataclasses` for data objects, `@property` for computed fields
- `@staticmethod` for helper methods that don't need instance state
- `async def` for all Flet event handlers and I/O operations

## Naming
- `snake_case` for variables, functions, methods, modules
- `PascalCase` for classes
- `UPPER_SNAKE` for module-level constants
- Private methods prefixed with `_`, use `__` only for name mangling

## Project patterns
- CVM layers: Model (`models/`) → Controller (`controllers/`) → View (`views/`)
- Callbacks flow Controller → View: `on_*` setters, View wires them in `_wire_callbacks()`
- Controller owns runtime state; View reads state via `page.update()` after mutation
- Logger per module: `logger = logging.getLogger(__name__)`
- Error display: Controller emits via `_emit_error()`, View shows in status text
- Single `pyproject.toml`; no `__init__.py` boilerplate beyond re-exports
