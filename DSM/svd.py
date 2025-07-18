
import numpy as np

A = np.array([[3, 1],
              [1, 3]])

print("Original matrix A:\n", A)

U, S, VT = np.linalg.svd(A)

print("\nMatrix U (Left singular vectors):\n", U)
print("\nSingular values (Sigma):\n", S)
print("\nMatrix V^T (Right singular vectors transpose):\n", VT)


Sigma = np.zeros((A.shape[0], A.shape[1]))  
np.fill_diagonal(Sigma, S)

print("\nDiagonal matrix Sigma:\n", Sigma)


A_reconstructed = U @ Sigma @ VT

print("\nReconstructed matrix A (U * Sigma * V^T):\n", np.round(A_reconstructed, 2))
