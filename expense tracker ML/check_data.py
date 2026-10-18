import pandas as pd

df = pd.read_csv("personal_finance_dataset.csv")

print("Dataset loaded successfully!")
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDate range:")
df["date"] = pd.to_datetime(df["date"])
print("Start date:", df["date"].min())
print("End date:", df["date"].max())

print("\nTransactions by type:")
print(df["income_or_expense"].value_counts())

print("\nUnique months:")
print(df[["year", "month_number"]].drop_duplicates().shape[0])

print("\nExpense categories:")
print(df["category"].value_counts())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nAmount summary:")
print(df["amount"].describe())
