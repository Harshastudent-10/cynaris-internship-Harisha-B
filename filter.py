import pandas as pd

# Load dataset
df = pd.read_csv("districts.csv")

# Display basic information
print("Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nFirst 10 rows:")
print(df.head(10))

# FILTER
# Select Karnataka records
filtered_df = df[df["State"] == "Karnataka"]

print("\nFiltered Data:")
print(filtered_df)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Create Year-Month column
df["YearMonth"] = df["Date"].dt.to_period("M")

print("\nYear-Month:")
print(df[["Date", "YearMonth"]].head())