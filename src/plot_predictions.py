import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
# FEATURES
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

X = df[features]
y = df["Close"]

# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df["Date"].iloc[split_index:]

# ==========================================
# SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    test_dates,
    y_test,
    label="Actual Close Price"
)

plt.plot(
    test_dates,
    y_pred,
    label="Predicted Close Price"
)

plt.xlabel("Date")

plt.ylabel("Close Price")

plt.title("YES Bank Stock Price: Actual vs Predicted")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()