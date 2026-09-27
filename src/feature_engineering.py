import pandas as pd

# Load dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# ==========================================
# CREATE LEAKAGE-FREE TECHNICAL FEATURES
# ==========================================

# Previous month's closing price
df["Previous_Close"] = df["Close"].shift(1)

# Previous month's return
df["Return"] = df["Close"].pct_change().shift(1)

# 3-month moving average using only previous closing prices
df["MA_3"] = df["Close"].shift(1).rolling(window=3).mean()

# 3-month volatility using only previous returns
df["Volatility_3"] = df["Return"].rolling(window=3).std()


# ==========================================
# DISPLAY TECHNICAL FEATURES
# ==========================================

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


# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

print(
    df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Previous_Close",
            "Return",
            "MA_3",
            "Volatility_3"
        ]
    ].isnull().sum()
)


# ==========================================
# REMOVE ROWS WITH MISSING VALUES
# ==========================================

df = df.dropna().reset_index(drop=True)

print("\n========== AFTER REMOVING MISSING VALUES ==========")

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
    ].head()
)

print("\nDataset shape:", df.shape)