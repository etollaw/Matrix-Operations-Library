"""Flask web application — Matrix Operations Calculator.

Routes
------
GET  /             → Serve the calculator UI.
POST /api/compute  → JSON API for matrix computations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

import matrix_operations as mo
from matrix_operations.operations import (
    InvalidInputError,
    MatrixError,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_matrix(text: str) -> list[list[float]]:
    """Parse a matrix string like ``'1 2; 3 4'`` into a list-of-lists.

    Rows can be separated by newlines or semicolons.
    Values within a row are separated by whitespace or commas.
    """
    text = text.strip()
    if not text:
        raise InvalidInputError("Matrix input is empty")

    rows = [r.strip() for r in text.replace("\n", ";").split(";") if r.strip()]
    matrix: list[list[float]] = []
    col_count: int | None = None

    for i, row_str in enumerate(rows):
        tokens = row_str.replace(",", " ").split()
        values: list[float] = []
        for tok in tokens:
            try:
                values.append(float(tok))
            except ValueError:
                raise InvalidInputError(
                    f"Row {i + 1}: '{tok}' is not a valid number"
                )
        if col_count is None:
            col_count = len(values)
        elif len(values) != col_count:
            raise InvalidInputError(
                f"Row {i + 1} has {len(values)} values but row 1 has {col_count}"
            )
        matrix.append(values)
    return matrix


def _parse_vector(text: str) -> list[float]:
    """Parse a vector string like ``'1 2 3'`` into a list of floats."""
    text = text.strip()
    if not text:
        raise InvalidInputError("Vector input is empty")
    tokens = text.replace(";", " ").replace("\n", " ").replace(",", " ").split()
    vector: list[float] = []
    for tok in tokens:
        try:
            vector.append(float(tok))
        except ValueError:
            raise InvalidInputError(f"'{tok}' is not a valid number")
    return vector


def _jsonable(obj: Any) -> Any:
    """Recursively convert NumPy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.complexfloating)):
        val = complex(obj)
        return val.real if val.imag == 0 else {"re": val.real, "im": val.imag}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, complex):
        return obj.real if obj.imag == 0 else {"re": obj.real, "im": obj.imag}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the calculator UI."""
    return render_template("index.html")


@app.route("/api/compute", methods=["POST"])
def compute():
    """JSON API for matrix computations.

    Expects JSON body::

        {
            "operation": "multiply",
            "matrix_a": "1 2; 3 4",
            "matrix_b": "5 6; 7 8",
            "vector_b": "1 2"
        }
    """
    try:
        data: dict[str, Any] = request.get_json(force=True)
        operation = data.get("operation", "").strip().lower()

        # Matrix A is always required
        matrix_a = _parse_matrix(data.get("matrix_a", ""))

        result: dict[str, Any] = {"operation": operation, "success": True}

        if operation == "multiply":
            matrix_b = _parse_matrix(data.get("matrix_b", ""))
            product = mo.multiply(matrix_a, matrix_b)
            result["result"] = _jsonable(product)
            result["description"] = (
                f"Product of {len(matrix_a)}×{len(matrix_a[0])} "
                f"and {len(matrix_b)}×{len(matrix_b[0])} matrices"
            )

        elif operation == "determinant":
            d = mo.det(matrix_a)
            result["result"] = round(d, 10)
            result["description"] = (
                f"Determinant of {len(matrix_a)}×{len(matrix_a[0])} matrix"
            )

        elif operation == "inverse":
            inv = mo.inverse(matrix_a)
            result["result"] = _jsonable(inv)
            result["description"] = (
                f"Inverse of {len(matrix_a)}×{len(matrix_a[0])} matrix"
            )

        elif operation == "transpose":
            t = mo.transpose(matrix_a)
            result["result"] = _jsonable(t)
            result["description"] = (
                f"Transpose: {len(matrix_a)}×{len(matrix_a[0])} → "
                f"{len(matrix_a[0])}×{len(matrix_a)}"
            )

        elif operation == "eigen":
            values, vectors = mo.eig(matrix_a)
            result["result"] = {
                "eigenvalues": _jsonable(values),
                "eigenvectors": _jsonable(vectors),
            }
            result["description"] = (
                f"Eigenvalues and eigenvectors of "
                f"{len(matrix_a)}×{len(matrix_a[0])} matrix"
            )

        elif operation == "solve":
            vector_b = _parse_vector(data.get("vector_b", ""))
            x = mo.solve(matrix_a, vector_b)
            result["result"] = _jsonable(x)
            result["description"] = (
                f"Solution to Ax = b where A is "
                f"{len(matrix_a)}×{len(matrix_a[0])}"
            )

        elif operation == "lu":
            P, L, U = mo.lu(matrix_a)
            result["result"] = {
                "P": _jsonable(P),
                "L": _jsonable(L),
                "U": _jsonable(U),
            }
            result["description"] = (
                f"LU decomposition of {len(matrix_a)}×{len(matrix_a[0])} matrix"
            )

        elif operation == "rank":
            r = mo.rank(matrix_a)
            result["result"] = r
            result["description"] = (
                f"Rank of {len(matrix_a)}×{len(matrix_a[0])} matrix"
            )

        elif operation == "trace":
            t = mo.trace(matrix_a)
            result["result"] = round(t, 10)
            result["description"] = (
                f"Trace of {len(matrix_a)}×{len(matrix_a[0])} matrix"
            )

        else:
            return (
                jsonify({"success": False, "error": f"Unknown operation: '{operation}'"}),
                400,
            )

        return jsonify(result)

    except MatrixError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Unexpected error: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
