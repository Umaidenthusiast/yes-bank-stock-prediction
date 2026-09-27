import pandas as pd

# Load dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# ==========================================
# TECHNICAL FEATURES
# ==========================================

# Previous closing price
df["Previous_Close"] = df["Close"].shift(1)

# Monthly return
df["Return"] = df["Close"].pct_change()

# 3-month moving average
df["MA_3"] = df["Close"].rolling(window=3).mean()

# 3-month volatility of returns
df["Volatility_3"] = df["Return"].rolling(window=3).std()

print("========== TECHNICAL FEATURES ==========")

print(
    df[
        [
            "Date",
            "Close",
            "Previous_Close",
            "Return",
            "MA_3",
            "Volatility_3"
        ]
    ].head(10)
)

print("\n========== MISSING VALUES ==========")

print(
    df[
        [
            "Previous_Close",
            "Return",
            "MA_3",
            "Volatility_3"
        ]
    ].isnull().sum()
)

# Remove rows created by rolling calculations
df_clean = df.dropna().reset_index(drop=True)

print("\n========== AFTER REMOVING MISSING VALUES ==========")

print(
    df_clean[
        [
            "Date",
            "Close",
            "Previous_Close",
            "Return",
            "MA_3",
            "Volatility_3"
        ]
    ].head()
)

print("\nDataset shape:", df_clean.shape)