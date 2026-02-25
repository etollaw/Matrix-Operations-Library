"""Matrix operations library — validated, typed wrappers over NumPy / SciPy.

Every public function converts its input to a NumPy array, validates
dimensions, and raises a descriptive custom exception on failure.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.linalg

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Matrix = Union[list[list[float]], np.ndarray]
Vector = Union[list[float], np.ndarray]

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MatrixError(Exception):
    """Base exception for matrix operation errors."""


class DimensionError(MatrixError):
    """Raised when matrix dimensions are incompatible for the operation."""


class SingularMatrixError(MatrixError):
    """Raised when a non-singular matrix is required but the matrix is singular."""


class InvalidInputError(MatrixError):
    """Raised when input cannot be interpreted as a valid numeric matrix."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_array(m: Matrix, name: str = "matrix") -> np.ndarray:
    """Convert *m* to a 2-D ``float64`` NumPy array with validation.

    Parameters
    ----------
    m : Matrix
        A list-of-lists or ndarray representing a matrix.
    name : str
        Label used in error messages (e.g. ``"A"``).

    Raises
    ------
    InvalidInputError
        If *m* is empty, non-numeric, or not 2-D.
    """
    try:
        arr = np.asarray(m, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise InvalidInputError(
            f"{name}: cannot convert to numeric matrix — {exc}"
        ) from exc

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise InvalidInputError(
            f"{name}: expected a 2-D matrix, got {arr.ndim}-D array"
        )
    if arr.size == 0:
        raise InvalidInputError(f"{name}: matrix must not be empty")
    return arr


def _to_vector(v: Vector, name: str = "vector") -> np.ndarray:
    """Convert *v* to a 1-D ``float64`` NumPy array with validation."""
    try:
        arr = np.asarray(v, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise InvalidInputError(
            f"{name}: cannot convert to numeric vector — {exc}"
        ) from exc
    arr = arr.flatten()
    if arr.size == 0:
        raise InvalidInputError(f"{name}: vector must not be empty")
    return arr


def _validate_square(a: np.ndarray, op: str) -> None:
    """Raise :class:`DimensionError` if *a* is not square."""
    if a.shape[0] != a.shape[1]:
        raise DimensionError(
            f"{op} requires a square matrix, got shape "
            f"{a.shape[0]}×{a.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def multiply(a: Matrix, b: Matrix) -> np.ndarray:
    """Multiply two matrices ``A × B``.

    Parameters
    ----------
    a : Matrix — shape (m, n)
    b : Matrix — shape (n, p)

    Returns
    -------
    np.ndarray — shape (m, p)

    Raises
    ------
    DimensionError
        If the inner dimensions do not match.
    """
    A = _to_array(a, "A")
    B = _to_array(b, "B")
    if A.shape[1] != B.shape[0]:
        raise DimensionError(
            f"Cannot multiply: A is {A.shape[0]}×{A.shape[1]} but B is "
            f"{B.shape[0]}×{B.shape[1]} — inner dimensions "
            f"{A.shape[1]} ≠ {B.shape[0]}"
        )
    return A @ B


def det(a: Matrix) -> float:
    """Compute the determinant of a square matrix.

    Parameters
    ----------
    a : Matrix — must be square (n × n)

    Returns
    -------
    float

    Raises
    ------
    DimensionError
        If the matrix is not square.
    """
    A = _to_array(a, "A")
    _validate_square(A, "Determinant")
    return float(np.linalg.det(A))


def inverse(a: Matrix) -> np.ndarray:
    """Compute the inverse of a square non-singular matrix.

    Parameters
    ----------
    a : Matrix — must be square and non-singular

    Returns
    -------
    np.ndarray

    Raises
    ------
    DimensionError
        If the matrix is not square.
    SingularMatrixError
        If the matrix is singular (det ≈ 0).
    """
    A = _to_array(a, "A")
    _validate_square(A, "Inverse")
    d = np.linalg.det(A)
    if np.isclose(d, 0.0, atol=1e-12):
        raise SingularMatrixError(
            f"Matrix is singular (det ≈ {d:.2e}); inverse does not exist"
        )
    return np.linalg.inv(A)


def transpose(a: Matrix) -> np.ndarray:
    """Transpose a matrix (swap rows ↔ columns).

    Parameters
    ----------
    a : Matrix — any shape (m × n)

    Returns
    -------
    np.ndarray — shape (n × m)
    """
    return _to_array(a, "A").T


def eig(a: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and right eigenvectors of a square matrix.

    Parameters
    ----------
    a : Matrix — must be square (n × n)

    Returns
    -------
    (eigenvalues, eigenvectors)
        eigenvalues  — 1-D array of length *n* (may be complex)
        eigenvectors — 2-D array; column *i* corresponds to eigenvalue *i*

    Raises
    ------
    DimensionError
        If the matrix is not square.
    """
    A = _to_array(a, "A")
    _validate_square(A, "Eigenvalue decomposition")
    values, vectors = np.linalg.eig(A)
    return values, vectors


def solve(a: Matrix, b: Vector) -> np.ndarray:
    """Solve the linear system ``A · x = b``.

    Parameters
    ----------
    a : Matrix — square (n × n) coefficient matrix
    b : Vector — right-hand side of length *n*

    Returns
    -------
    np.ndarray — solution vector *x* of length *n*

    Raises
    ------
    DimensionError
        If *A* is not square or *b* length does not match.
    SingularMatrixError
        If *A* is singular.
    """
    A = _to_array(a, "A")
    bv = _to_vector(b, "b")
    _validate_square(A, "Solve Ax=b")
    if A.shape[0] != bv.shape[0]:
        raise DimensionError(
            f"Dimension mismatch: A is {A.shape[0]}×{A.shape[1]} "
            f"but b has length {bv.shape[0]}"
        )
    d = np.linalg.det(A)
    if np.isclose(d, 0.0, atol=1e-12):
        raise SingularMatrixError(
            f"Coefficient matrix is singular (det ≈ {d:.2e}); "
            "system may have no solution or infinitely many solutions"
        )
    return np.linalg.solve(A, bv)


def lu(a: Matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the PLU decomposition so that ``P · A = L · U``.

    Uses SciPy's LU with partial pivoting.  SciPy returns ``P, L, U``
    such that ``A = P @ L @ U`` (P is the permutation matrix).

    Parameters
    ----------
    a : Matrix — shape (m × n)

    Returns
    -------
    (P, L, U)
        P — permutation matrix (m × m)
        L — lower-triangular (m × k), k = min(m, n)
        U — upper-triangular (k × n)
    """
    A = _to_array(a, "A")
    P, L, U = scipy.linalg.lu(A)
    return P, L, U


def rank(a: Matrix) -> int:
    """Compute the rank of a matrix.

    Parameters
    ----------
    a : Matrix

    Returns
    -------
    int
    """
    return int(np.linalg.matrix_rank(_to_array(a, "A")))


def trace(a: Matrix) -> float:
    """Compute the trace (sum of diagonal elements) of a square matrix.

    Parameters
    ----------
    a : Matrix — must be square

    Returns
    -------
    float

    Raises
    ------
    DimensionError
        If the matrix is not square.
    """
    A = _to_array(a, "A")
    _validate_square(A, "Trace")
    return float(np.trace(A))
