import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# ==========================================
# CREATE TECHNICAL FEATURES
# ==========================================

df["Previous_Close"] = df["Close"].shift(1)

df["Return"] = df["Close"].pct_change()

df["MA_3"] = df["Close"].rolling(window=3).mean()

df["Volatility_3"] = df["Return"].rolling(window=3).std()
# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

# ==========================================
# FEATURES AND TARGET
# ==========================================

features = [
    "Open",
    "High",
    "Low",
    "Previous_Close",
    "Return",
    "MA_3",
    "Volatility_3"
]

target = "Close"

X = df[features]
y = df[target]

# ==========================================
# CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

print("========== TECHNICAL FEATURE PREPARATION ==========")

print("Features:")
for feature in features:
    print("-", feature)

print("\nTotal samples:", len(df))

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\n========== TRAINING DATE RANGE ==========")
print("First:", df["Date"].iloc[0])
print("Last:", df["Date"].iloc[len(X_train) - 1])

print("\n========== TESTING DATE RANGE ==========")
print("First:", df["Date"].iloc[len(X_train)])
print("Last:", df["Date"].iloc[-1])

print("\n========== FIRST 5 TRAINING ROWS ==========")
print(X_train.head())

print("\n========== FIRST 5 TESTING ROWS ==========")
print(X_test.head())