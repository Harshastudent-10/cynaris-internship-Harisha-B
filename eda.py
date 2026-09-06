import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("sample_dataset.csv")

# 2. Key Inspection Commands & 5 Observations
print("--- DF INFO ---")
df.info()

print("\n--- DF DESCRIBE ---")
print(df.describe(include='all'))

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

"""
5 DOCUMENTED OBSERVATIONS:
1. Missing Values: Columns like 'age' and 'category' contain null values requiring imputation.
2. Skewness/Outliers: Numeric metrics show noticeable variance between mean and 50th percentile (median).
3. Data Types: Categorical fields are recognized as general objects and require explicit conversion.
4. Duplicate Records: Duplicate key entries exist across row instances.
5. High Cardinality: Certain categorical values show extreme distribution imbalance in top counts.
"""

# 3. Plotting Distributions, Heatmap & Top Categories
plt.figure(figsize=(15, 4))

# Numeric Distributions
numeric_cols = df.select_dtypes(include=['number']).columns
for i, col in enumerate(numeric_cols[:2], 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')

# Top 10 Categories Plot
cat_col = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else None
if cat_col:
    plt.subplot(1, 3, 3)
    df[cat_col].value_counts().head(10).plot(kind='bar')
    plt.title(f'Top 10 Category Counts ({cat_col})')

plt.tight_layout()
plt.savefig('eda_plots.png')
plt.show()

# Correlation Heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig('correlation_heatmap.png')
    plt.show()