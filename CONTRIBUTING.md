# Contributing to Matrix Operations Library

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/etollaw/Matrix-Operations-Library.git
cd Matrix-Operations-Library

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install in dev mode
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v
```

## Linting

```bash
ruff check .
ruff check . --fix   # auto-fix
```

## Workflow

1. Fork the repo and create a feature branch from `main`.
2. Make your changes with tests.
3. Run `pytest -v` and `ruff check .` — both must pass.
4. Open a Pull Request with a clear description.

## Adding a New Operation

1. Add the function to `matrix_operations/operations.py` with full docstring,
   type hints, and input validation.
2. Export it in `matrix_operations/__init__.py`.
3. Add the operation to `web/app.py` (both the API handler and the frontend dropdown).
4. Write tests in `tests/test_operations.py`.

## Code Style

- Type hints on all public functions.
- Docstrings in NumPy-style.
- Keep functions focused — one operation per function.
- Validate inputs and raise the appropriate custom exception
  (`DimensionError`, `SingularMatrixError`, `InvalidInputError`).
