<div align="center">

# 🧮 Matrix Operations Library

**A Python library + web calculator for linear algebra operations.**
<br/>Determinants · Inverses · Eigenvalues · Solve Ax=b · LU Decomposition — and more.

[![CI](https://github.com/etollaw/Matrix-Operations-Library/actions/workflows/ci.yml/badge.svg)](https://github.com/etollaw/Matrix-Operations-Library/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com)

[**Try the Live Demo →**](#web-calculator) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Contributing](CONTRIBUTING.md)

</div>

---

## ✨ Features

| Operation | Function | Input | Notes |
|:---|:---|:---|:---|
| **Multiplication** | `mo.multiply(A, B)` | Two matrices | Validates inner dimensions |
| **Determinant** | `mo.det(A)` | Square matrix | Returns scalar |
| **Inverse** | `mo.inverse(A)` | Square, non-singular | Detects singular matrices |
| **Transpose** | `mo.transpose(A)` | Any matrix | Swaps rows ↔ columns |
| **Eigenvalues** | `mo.eig(A)` | Square matrix | Returns (values, vectors) |
| **Solve Ax = b** | `mo.solve(A, b)` | Square A + vector b | Checks solvability |
| **LU Decomposition** | `mo.lu(A)` | Any matrix | Returns P, L, U |
| **Rank** | `mo.rank(A)` | Any matrix | Returns integer rank |
| **Trace** | `mo.trace(A)` | Square matrix | Sum of diagonal |

All operations include **type hints**, **input validation**, and **clear error messages**.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/etollaw/Matrix-Operations-Library.git
cd Matrix-Operations-Library
pip install -e ".[dev]"
```

### Library Usage

```python
import matrix_operations as mo

# Multiply two matrices
C = mo.multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
# → array([[19, 22], [43, 50]])

# Determinant
d = mo.det([[1, 2], [3, 4]])   # → -2.0

# Inverse
inv = mo.inverse([[4, 7], [2, 6]])

# Eigenvalues & eigenvectors
vals, vecs = mo.eig([[2, 1], [1, 2]])

# Solve Ax = b
x = mo.solve([[2, 1], [5, 3]], [4, 7])

# LU decomposition
P, L, U = mo.lu([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

# Rank and trace
r = mo.rank([[1, 2], [2, 4]])  # → 1
t = mo.trace([[1, 0], [0, 1]]) # → 2.0
```

### Error Handling

```python
from matrix_operations import DimensionError, SingularMatrixError

try:
    mo.inverse([[1, 2], [2, 4]])  # singular!
except SingularMatrixError as e:
    print(e)  # "Matrix is singular (det ≈ 0.00e+00); inverse does not exist"

try:
    mo.multiply([[1, 2]], [[1, 2]])  # dimension mismatch
except DimensionError as e:
    print(e)  # "Cannot multiply: A is 1×2 but B is 1×2 — inner dimensions 2 ≠ 1"
```

---

## 🌐 Web Calculator

A browser-based matrix calculator powered by Flask.

**Features:**
- Textarea input — enter matrices row-by-row or with semicolons
- Operation dropdown with all 9 operations
- Pre-loaded example matrices (one-click load)
- Matrix-bracket styled result rendering
- Error messages for dimension mismatches, singular matrices, invalid input
- Copy result to clipboard
- Mobile responsive

### Run Locally

```bash
python web/app.py
# → http://localhost:5000
```

### Deploy to Vercel

The repo is pre-configured for **one-click Vercel deployment**:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fetollaw%2FMatrix-Operations-Library)

Or deploy manually:

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### JSON API

All computation goes through a single endpoint:

**`POST /api/compute`**

```json
{
    "operation": "multiply",
    "matrix_a": "1 2; 3 4",
    "matrix_b": "5 6; 7 8"
}
```

**Response:**

```json
{
    "success": true,
    "operation": "multiply",
    "result": [[19, 22], [43, 50]],
    "description": "Product of 2×2 and 2×2 matrices"
}
```

<details>
<summary><strong>All supported operations</strong></summary>

| Operation | Required fields | Optional |
|:---|:---|:---|
| `multiply` | `matrix_a`, `matrix_b` | — |
| `determinant` | `matrix_a` | — |
| `inverse` | `matrix_a` | — |
| `transpose` | `matrix_a` | — |
| `eigen` | `matrix_a` | — |
| `solve` | `matrix_a`, `vector_b` | — |
| `lu` | `matrix_a` | — |
| `rank` | `matrix_a` | — |
| `trace` | `matrix_a` | — |

Matrix input format: space-separated values, rows split by `;` or newlines.
Example: `"1 2 3; 4 5 6; 7 8 9"`

</details>

---

## 🧪 Tests

**34 unit tests** covering correctness, edge cases, error handling, and NumPy cross-validation.

```bash
pytest -v
```

```
tests/test_operations.py::TestMultiply::test_2x2                 PASSED
tests/test_operations.py::TestMultiply::test_identity             PASSED
tests/test_operations.py::TestMultiply::test_rectangular          PASSED
tests/test_operations.py::TestMultiply::test_dimension_mismatch   PASSED
tests/test_operations.py::TestDeterminant::test_3x3              PASSED
tests/test_operations.py::TestInverse::test_2x2_roundtrip        PASSED
tests/test_operations.py::TestSolve::test_singular_raises         PASSED
...
============================= 34 passed ==============================
```

---

## 📁 Project Structure

```
├── matrix_operations/         # 📦 Importable Python package
│   ├── __init__.py            #    Public API + exports
│   └── operations.py          #    All 9 operations + validation
├── web/                       # 🌐 Flask web calculator
│   ├── app.py                 #    Routes + JSON API
│   └── templates/
│       └── index.html         #    Calculator SPA
├── api/                       # ▲ Vercel serverless adapter
│   └── index.py
├── tests/                     # 🧪 Pytest suite (34 tests)
│   └── test_operations.py
├── .github/workflows/
│   └── ci.yml                 # ⚙️ GitHub Actions CI
├── vercel.json                # ▲ Vercel deployment config
├── pyproject.toml             # 📋 Package config + dependencies
├── requirements.txt           # 📋 Vercel dependency list
├── CONTRIBUTING.md            # 🤝 Contributor guide
└── LICENSE                    # MIT
```

---

## 🏗️ Architecture

```
┌─────────────────────────────┐     ┌──────────────────────┐
│  Web UI  (index.html)       │     │  Your Python code    │
│  ┌───────────────────────┐  │     │                      │
│  │ Textarea → JSON body  │──┼──→  │  import matrix_ops   │
│  │ fetch('/api/compute') │  │     │  mo.multiply(A, B)   │
│  └───────────────────────┘  │     │  mo.det(A)           │
└─────────────────────────────┘     └──────────────────────┘
              │                                │
              ▼                                ▼
┌─────────────────────────────┐     ┌──────────────────────┐
│  Flask API  (app.py)        │     │  matrix_operations/  │
│  POST /api/compute          │────→│  operations.py       │
│  Parse → Validate → Compute │     │  NumPy + SciPy       │
│  Return JSON                │     └──────────────────────┘
└─────────────────────────────┘
```

---

## 🔧 Development

```bash
# Lint
ruff check .

# Lint + auto-fix
ruff check . --fix

# Run tests
pytest -v

# Run web app in dev mode
python web/app.py
```

---

## 📖 API Reference

### Custom Exceptions

| Exception | When |
|:---|:---|
| `MatrixError` | Base exception for all matrix errors |
| `DimensionError` | Incompatible dimensions for the operation |
| `SingularMatrixError` | Non-singular matrix required but matrix is singular |
| `InvalidInputError` | Input cannot be parsed as a valid numeric matrix |

### Dependencies

| Package | Purpose |
|:---|:---|
| **NumPy** | Core numerical computation |
| **SciPy** | LU decomposition |
| **Flask** | Web framework |
| **pytest** | Testing (dev) |
| **ruff** | Linting (dev) |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with NumPy, SciPy, and Flask · Made by <a href="https://github.com/etollaw">@etollaw</a></sub>
</div>
