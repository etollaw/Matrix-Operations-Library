"""Matrix Operations Library — efficient, validated matrix computations.

Usage::

    import matrix_operations as mo

    C = mo.multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    d = mo.det([[1, 2], [3, 4]])
"""

from matrix_operations.operations import (
    DimensionError,
    InvalidInputError,
    # Exceptions
    MatrixError,
    SingularMatrixError,
    det,
    eig,
    inverse,
    lu,
    # Functions
    multiply,
    rank,
    solve,
    trace,
    transpose,
)

__all__ = [
    "multiply",
    "det",
    "inverse",
    "transpose",
    "eig",
    "solve",
    "lu",
    "rank",
    "trace",
    "MatrixError",
    "DimensionError",
    "SingularMatrixError",
    "InvalidInputError",
]

__version__ = "1.0.0"
