import pandas as pd

df = pd.read_csv("districts.csv")

print(df.shape)          # (257905, 8)
print(df.dtypes)
print(df.head(10))