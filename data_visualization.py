import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(df: pd.DataFrame) -> None:
    """Generates and saves standard EDA plots using Matplotlib and Seaborn."""
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution Plot (Histogram + KDE)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[numeric_cols[0]].dropna(), kde=True, color="skyblue")
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel("Frequency")
        plt.savefig("distribution_plot.png")
        plt.close()

    # 2. Category Counts (Bar Plot)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        plt.figure(figsize=(8, 5))
        top_cats = df[categorical_cols[0]].value_counts().head(10)
        sns.barplot(x=top_cats.index, y=top_cats.values, palette="viridis")
        plt.title(f"Top Categories in {categorical_cols[0]}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("category_counts.png")
        plt.close()

    # 3. Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

if __name__ == "__main__":
    # Create sample dataset if CSV does not exist
    if not os.path.exists("sample_dataset.csv"):
        data = {
            "age": [25, 30, 35, 40, 22, 28, 32, 45, 50, 29],
            "salary": [50000, 60000, 70000, 80000, 45000, 58000, 65000, 90000, 110000, 62000],
            "department": ["IT", "HR", "IT", "Sales", "HR", "IT", "Sales", "IT", "Sales", "HR"]
        }
        pd.DataFrame(data).to_csv("sample_dataset.csv", index=False)

    df = pd.read_csv("sample_dataset.csv")
    generate_visualizations(df)
    print("Visualizations generated and saved successfully.")