import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Create previous close feature
df["Previous_Close"] = df["Close"].shift(1)

# Remove first row
df = df.dropna().reset_index(drop=True)

# Define features and target
features = ["Open", "High", "Low", "Previous_Close"]
target = "Close"

X = df[features]
y = df[target]

# Chronological train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

print("========== FEATURE PREPARATION ==========")

print("Features:")
print(features)

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