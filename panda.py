import pandas as pd
import numpy as np

# Option A – load a real CSV (replace with your file)
# df = pd.read_csv("your_dataset.csv")

# Option B – quick demo with Iris-like data
data = {
    "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0],
    "sepal_width":  [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4],
    "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5],
    "petal_width":  [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2]
}
df = pd.DataFrame(data)

# Convert to NumPy
arr = df.to_numpy()

# Mean
print("Column-wise mean:", np.mean(arr, axis=0))

# Standard deviation
print("Column-wise std :", np.std(arr, axis=0))

# Correlation matrix
corr = np.corrcoef(arr, rowvar=False)
print("\nCorrelation matrix:\n", np.round(corr, 3))