# Broadcasting
import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])           # shape (3,) broadcasts to (2, 3)
print("Broadcasting result:\n", a + b)

# Vectorised operations
x = np.arange(1, 11)
print("\nSquared (vectorised):", x ** 2)
print("Sin values:", np.sin(x))
print("Cumulative sum:", np.cumsum(x))

# Matrix multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print("\nMatrix multiplication (A @ B):\n", A @ B)
# or np.matmul(A, B) / np.dot(A, B)