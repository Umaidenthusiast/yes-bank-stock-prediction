import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Remove missing values
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
# CHRONOLOGICAL SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

# ==========================================
# STANDARD SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# OUTPUT
# ==========================================

print("========== TECHNICAL FEATURE SCALING ==========")

print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

print("\n========== FIRST 5 SCALED TRAINING ROWS ==========")
print(X_train_scaled[:5])

print("\n========== FIRST 5 SCALED TESTING ROWS ==========")
print(X_test_scaled[:5])

print("\n========== TRAINING SCALED MEAN ==========")
print(X_train_scaled.mean(axis=0))

print("\n========== TRAINING SCALED STD ==========")
print(X_train_scaled.std(axis=0))