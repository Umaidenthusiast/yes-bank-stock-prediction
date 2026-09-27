import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("data/yes_bank_stock.csv")

data["Date"] = pd.to_datetime(data["Date"])


# ==========================================
# CREATE TECHNICAL FEATURES
# ==========================================

data["Previous_Close"] = data["Close"].shift(1)

data["Return"] = data["Close"].pct_change()

data["MA_3"] = data["Close"].rolling(window=3).mean()

data["Volatility_3"] = data["Return"].rolling(window=3).std()


# ==========================================
# REMOVE MISSING VALUES
# ==========================================

data = data.dropna().reset_index(drop=True)


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

X = data[features]
y = data["Close"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

dates_train = data["Date"].iloc[:split_index]
dates_test = data["Date"].iloc[split_index:]


# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# TRAIN LINEAR REGRESSION V3
# ==========================================

model = LinearRegression()

model.fit(X_train_scaled, y_train)


# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_scaled)


# ==========================================
# CALCULATE RESIDUALS
# ==========================================

residuals = y_test.values - y_pred


# ==========================================
# RESIDUAL DATAFRAME
# ==========================================

residual_data = pd.DataFrame({
    "Date": dates_test.values,
    "Actual_Close": y_test.values,
    "Predicted_Close": y_pred,
    "Residual": residuals
})


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("========== RESIDUAL ANALYSIS ==========")

print("\nFirst 10 Residuals:")
print(residual_data.head(10))


print("\n========== RESIDUAL STATISTICS ==========")

print(f"Mean Residual: {residuals.mean():.4f}")
print(f"Minimum Residual: {residuals.min():.4f}")
print(f"Maximum Residual: {residuals.max():.4f}")
print(f"Residual Std Dev: {residuals.std():.4f}")


# ==========================================
# MODEL ERROR CHECK
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)


print("\n========== MODEL ERROR ==========")

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# ==========================================
# RESIDUAL VS PREDICTED PLOT
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Close Price")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted YES Bank Closing Price")

plt.grid(True)

plt.show()

# ==========================================
# RESIDUAL DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(residuals, bins=10, edgecolor="black")

plt.axvline(
    residuals.mean(),
    linestyle="--",
    label="Mean Residual"
)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Distribution of Linear Regression V3 Residuals")

plt.legend()
plt.grid(True)

plt.show()

# ==========================================
# RESIDUALS OVER TIME
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    dates_test,
    residuals,
    marker="o"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Date")
plt.ylabel("Residual")
plt.title("Residuals Over Time - Linear Regression V3")

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ==========================================
# SAVE FINAL MODEL AND SCALER
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/linear_regression_v3.pkl")

joblib.dump(scaler, "models/scaler_v3.pkl")

print("\n========== MODEL SAVING ==========")
print("Linear Regression V3 saved successfully!")
print("Model: models/linear_regression_v3.pkl")
print("Scaler: models/scaler_v3.pkl")