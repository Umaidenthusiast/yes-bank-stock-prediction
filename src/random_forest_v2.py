import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Create previous close feature
df["Previous_Close"] = df["Close"].shift(1)

# Remove first row
df = df.dropna().reset_index(drop=True)

# Features and target
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

# Create Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("========== RANDOM FOREST V2 ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\n========== MODEL EVALUATION ==========")

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Actual vs predicted
results = pd.DataFrame({
    "Date": df["Date"].iloc[len(X_train):].values,
    "Actual_Close": y_test.values,
    "Predicted_Close": y_pred
})

print("\n========== FIRST 10 PREDICTIONS ==========")
print(results.head(10))