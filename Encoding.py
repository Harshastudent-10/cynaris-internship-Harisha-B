import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder
)



df = pd.read_csv("customers-100.csv")

print("Dataset loaded successfully!")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())




le = LabelEncoder()

df["Country_encoded"] = le.fit_transform(df["Country"])

print("\n--- LABEL ENCODER ---")

print(
    df[["Country", "Country_encoded"]].head(10)
)




ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

city_encoded = ohe.fit_transform(
    df[["City"]]
)

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=ohe.get_feature_names_out(["City"]),
    index=df.index
)

print("\n--- ONE HOT ENCODER ---")

print(city_encoded_df.head())




country_order = sorted(
    df["Country"].dropna().unique()
)

oe = OrdinalEncoder(
    categories=[country_order],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

df["Country_ordinal"] = oe.fit_transform(
    df[["Country"]]
)

print("\n--- ORDINAL ENCODER ---")

print(
    df[["Country", "Country_ordinal"]].head(10)
)




print("\n================================")
print("ENCODING COMPLETED SUCCESSFULLY")
print("================================")

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal dataset:")
print(df.head())




df.to_csv(
    "customers-100-encoded.csv",
    index=False
)

print(
    "\nEncoded dataset saved as customers-100-encoded.csv"
)