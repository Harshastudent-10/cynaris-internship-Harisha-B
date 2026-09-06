# Create a virtual environment (run once in terminal)
# python -m venv .venv
# source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate             # Windows

# Install NumPy
# pip install numpy pandas

import numpy as np

# 1D array
arr_1d = np.array([10, 20, 30, 40, 50])
print("1D Array:", arr_1d)
print("Shape:", arr_1d.shape)          # (5,)

# 2D array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("\n2D Array:\n", arr_2d)
print("Shape:", arr_2d.shape)          # (3, 3)

# 3D array
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]],
                   [[9, 10], [11, 12]]])
print("\n3D Array:\n", arr_3d)
print("Shape:", arr_3d.shape)          # (3, 2, 2)