import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# TRAIN LINEAR REGRESSION
# ==========================================

model = LinearRegression()

model.fit(X_train_scaled, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_scaled)

# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("========== LINEAR REGRESSION V3 ==========")

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))

print("\n========== MODEL COEFFICIENTS ==========")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("\nIntercept:", round(model.intercept_, 4))

print("\n========== MODEL EVALUATION ==========")

print(f"MAE:  {mae:.4f}")

print(f"MSE:  {mse:.4f}")

print(f"RMSE: {rmse:.4f}")

print(f"R²:   {r2:.4f}")

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Date": df["Date"].iloc[len(X_train):].values,
    "Actual_Close": y_test.values,
    "Predicted_Close": y_pred
})

print("\n========== FIRST 10 PREDICTIONS ==========")

print(results.head(10))