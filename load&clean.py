import os
import pandas as pd
import mlflow
from typing import Tuple

# Configure MLflow tracking (optional local setup)
mlflow.set_experiment("Data_Loading_And_Cleaning")

def load_data(file_path: str) -> pd.DataFrame:
    """Loads dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[LOG] Data successfully loaded. Initial Shape: {df.shape}")
    return df

def inspect_data(df: pd.DataFrame) -> None:
    """Prints initial data structure, null counts, and summary stats."""
    print("\n--- DATA HEAD ---")
    print(df.head())
    
    print("\n--- DATA INFO ---")
    print(df.info())
    
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs missing value handling, deduplication, and type normalization."""
    df_clean = df.copy()
    
    # 1. Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"[LOG] Removed {initial_rows - len(df_clean)} duplicate rows.")
    
    # 2. Fill missing numeric values with median
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"[LOG] Filled missing values in numeric column '{col}' with median ({median_val}).")
            
    # 3. Fill missing categorical values with mode or default string
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"[LOG] Filled missing values in categorical column '{col}' with mode ({mode_val}).")
            
    return df_clean

def run_pipeline(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs data loading, inspection, and cleaning with MLflow logging."""
    with mlflow.start_run(run_name="Data_Preprocessing"):
        # Log input file parameter
        mlflow.log_param("data_path", data_path)
        
        # Load & Inspect
        raw_df = load_data(data_path)
        inspect_data(raw_df)
        
        # Clean
        cleaned_df = clean_data(raw_df)
        
        # Log metrics to MLflow
        mlflow.log_metric("raw_row_count", raw_df.shape[0])
        mlflow.log_metric("cleaned_row_count", cleaned_df.shape[0])
        mlflow.log_metric("column_count", cleaned_df.shape[1])
        
        # Save output artifact
        output_path = "cleaned_data.csv"
        cleaned_df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)
        
        print("\n[SUCCESS] Pipeline execution finished. Cleaned data saved and logged to MLflow.")
        return raw_df, cleaned_df

if __name__ == "__main__":
    # Example usage: Replace 'sample_dataset.csv' with your actual file path
    DATA_PATH = "sample_dataset.csv"
    
    # Create a small dummy CSV if it doesn't exist for immediate testing
    if not os.path.exists(DATA_PATH):
        sample_data = pd.DataFrame({
            "id": [1, 2, 2, 4, 5],
            "age": [25, 30, 30, None, 40],
            "category": ["A", "B", "B", "C", None]
        })
        sample_data.to_csv(DATA_PATH, index=False)
        print(f"[SETUP] Created mock file '{DATA_PATH}' for execution.")

    raw_data, clean_data_res = run_pipeline(DATA_PATH)