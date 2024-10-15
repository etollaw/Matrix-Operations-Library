import numpy as np

def matrix_multiplication(A, B):
    """Return the product of two matrices A and B."""
    return np.dot(A, B)

def determinant(A):
    """Return the determinant of matrix A."""
    return np.linalg.det(A)

def inverse_matrix(A):
    """Return the inverse of matrix A."""
    return np.linalg.inv(A)

def transpose_matrix(A):
    """Return the transpose of matrix A."""
    return np.transpose(A)

def eigenvalues_and_vectors(A):
    """Return the eigenvalues and eigenvectors of matrix A."""
    return np.linalg.eig(A)
