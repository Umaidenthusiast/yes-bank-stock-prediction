import pandas as pd

# Load the dataset
df = pd.read_csv("data/yes_bank_stock.csv")
# Convert Date column from string to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

print("\n========== DATE INFORMATION ==========")
print("First Date:", df["Date"].iloc[0])
print("Last Date:", df["Date"].iloc[-1])

print("\nFirst 10 Dates:")
print(df["Date"].head(10).to_list())

print("\nLast 10 Dates:")
print(df["Date"].tail(10).to_list())

print("\n========== DATE AFTER CONVERSION ==========")
print(df["Date"].head())
print("\nDate Data Type:", df["Date"].dtype)

print("\n========== NUMERICAL FEATURE ANALYSIS ==========")

numerical_columns = ["Open", "High", "Low", "Close"]

print(df[numerical_columns].describe().T)

print("\n========== SKEWNESS ANALYSIS ==========")

numerical_columns = ["Open", "High", "Low", "Close"]

for column in numerical_columns:
    print(f"{column}: {df[column].skew():.4f}")