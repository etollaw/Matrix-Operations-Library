"""Comprehensive unit tests for the matrix_operations library.

Each operation is tested for:
  - Correct output on simple inputs
  - Edge cases (singular, non-square, identity, …)
  - Proper exceptions on invalid input
  - Agreement with raw NumPy / SciPy for random inputs
"""

from __future__ import annotations

import numpy as np
import pytest

import matrix_operations as mo
from matrix_operations import (
    DimensionError,
    InvalidInputError,
    SingularMatrixError,
)

# ── Multiply ────────────────────────────────────────────────────────────────


class TestMultiply:
    def test_2x2(self):
        result = mo.multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        np.testing.assert_array_almost_equal(result, [[19, 22], [43, 50]])

    def test_identity(self):
        A = [[1, 2], [3, 4]]
        result = mo.multiply(A, [[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(result, A)

    def test_rectangular(self):
        A = [[1, 2, 3], [4, 5, 6]]      # 2×3
        B = [[7, 8], [9, 10], [11, 12]]  # 3×2
        result = mo.multiply(A, B)
        np.testing.assert_array_almost_equal(result, [[58, 64], [139, 154]])

    def test_dimension_mismatch_raises(self):
        with pytest.raises(DimensionError, match="inner dimensions"):
            mo.multiply([[1, 2], [3, 4]], [[1, 2, 3]])

    def test_against_numpy(self):
        rng = np.random.default_rng(42)
        A, B = rng.random((4, 3)), rng.random((3, 5))
        np.testing.assert_array_almost_equal(mo.multiply(A, B), A @ B)


# ── Determinant ─────────────────────────────────────────────────────────────


class TestDeterminant:
    def test_2x2(self):
        assert mo.det([[1, 2], [3, 4]]) == pytest.approx(-2.0)

    def test_identity(self):
        assert mo.det([[1, 0], [0, 1]]) == pytest.approx(1.0)

    def test_singular(self):
        assert mo.det([[1, 2], [2, 4]]) == pytest.approx(0.0, abs=1e-10)

    def test_3x3(self):
        assert mo.det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]) == pytest.approx(-306.0)

    def test_non_square_raises(self):
        with pytest.raises(DimensionError):
            mo.det([[1, 2, 3], [4, 5, 6]])

    def test_against_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        assert mo.det(A) == pytest.approx(np.linalg.det(A))


# ── Inverse ─────────────────────────────────────────────────────────────────


class TestInverse:
    def test_2x2_roundtrip(self):
        A = [[1, 2], [3, 4]]
        np.testing.assert_array_almost_equal(mo.multiply(A, mo.inverse(A)), np.eye(2))

    def test_singular_raises(self):
        with pytest.raises(SingularMatrixError):
            mo.inverse([[1, 2], [2, 4]])

    def test_non_square_raises(self):
        with pytest.raises(DimensionError):
            mo.inverse([[1, 2, 3], [4, 5, 6]])

    def test_against_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.random((4, 4))
        np.testing.assert_array_almost_equal(mo.inverse(A), np.linalg.inv(A))


# ── Transpose ───────────────────────────────────────────────────────────────


class TestTranspose:
    def test_2x2(self):
        np.testing.assert_array_almost_equal(
            mo.transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]]
        )

    def test_rectangular_shape(self):
        assert mo.transpose([[1, 2, 3], [4, 5, 6]]).shape == (3, 2)


# ── Eigenvalues ─────────────────────────────────────────────────────────────


class TestEig:
    def test_diagonal(self):
        vals, _ = mo.eig([[2, 0], [0, 3]])
        assert sorted(vals.real) == pytest.approx([2.0, 3.0])

    def test_non_square_raises(self):
        with pytest.raises(DimensionError):
            mo.eig([[1, 2, 3], [4, 5, 6]])

    def test_against_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.random((3, 3))
        vals, _ = mo.eig(A)
        np_vals, _ = np.linalg.eig(A)
        np.testing.assert_array_almost_equal(sorted(vals.real), sorted(np_vals.real))


# ── Solve ───────────────────────────────────────────────────────────────────


class TestSolve:
    def test_2x2(self):
        A, b = [[2, 1], [5, 3]], [4, 7]
        x = mo.solve(A, b)
        np.testing.assert_array_almost_equal(np.array(A) @ x, b)

    def test_3x3(self):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        b = [1, 2, 3]
        x = mo.solve(A, b)
        np.testing.assert_array_almost_equal(np.array(A) @ x, b)

    def test_singular_raises(self):
        with pytest.raises(SingularMatrixError):
            mo.solve([[1, 2], [2, 4]], [3, 6])

    def test_dimension_mismatch_raises(self):
        with pytest.raises(DimensionError):
            mo.solve([[1, 2], [3, 4]], [1, 2, 3])


# ── LU ──────────────────────────────────────────────────────────────────────


class TestLU:
    def test_2x2_reconstruction(self):
        A = [[2, 1], [4, 3]]
        P, L, U = mo.lu(A)
        np.testing.assert_array_almost_equal(P @ L @ U, A)

    def test_3x3_reconstruction(self):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        P, L, U = mo.lu(A)
        np.testing.assert_array_almost_equal(P @ L @ U, A)


# ── Rank ────────────────────────────────────────────────────────────────────


class TestRank:
    def test_full_rank(self):
        assert mo.rank([[1, 0], [0, 1]]) == 2

    def test_rank_deficient(self):
        assert mo.rank([[1, 2], [2, 4]]) == 1

    def test_rectangular(self):
        assert mo.rank([[1, 2, 3], [4, 5, 6]]) == 2


# ── Trace ───────────────────────────────────────────────────────────────────


class TestTrace:
    def test_2x2(self):
        assert mo.trace([[1, 2], [3, 4]]) == pytest.approx(5.0)

    def test_3x3_diagonal(self):
        assert mo.trace([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == pytest.approx(6.0)

    def test_non_square_raises(self):
        with pytest.raises(DimensionError):
            mo.trace([[1, 2, 3], [4, 5, 6]])


# ── Input validation ───────────────────────────────────────────────────────


class TestInputValidation:
    def test_empty_matrix(self):
        with pytest.raises(InvalidInputError):
            mo.det([])

    def test_non_numeric(self):
        with pytest.raises(InvalidInputError):
            mo.det([["a", "b"], ["c", "d"]])
