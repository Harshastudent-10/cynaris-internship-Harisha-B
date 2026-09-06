import os
import pandas as pd
df= pd.read_csv("districts.csv")

# Assuming 'df' is your cleaned DataFrame from previous steps

# 1. Export DataFrame to CSV
df.to_csv("cleaned_data.csv", index=False)

# 2. Export DataFrame to Parquet (requires pyarrow or fastparquet installed)
df.to_parquet("cleaned_data.parquet", index=False)

# 3. Get file sizes in bytes/KB
csv_size = os.path.getsize("cleaned_data.csv") / 1024  # Size in KB
parquet_size = os.path.getsize("cleaned_data.parquet") / 1024  # Size in KB

# 4. Print size comparison
print(f"CSV File Size: {csv_size:.2f} KB")
print(f"Parquet File Size: {parquet_size:.2f} KB")
print(
    f"Parquet is {csv_size / parquet_size:.2f}x smaller than CSV."
    if parquet_size > 0
    else ""
)