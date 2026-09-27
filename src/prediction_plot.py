import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Define features and target
features = ["Open", "High", "Low"]
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

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Test dates
test_dates = df["Date"].iloc[len(X_train):]

# Plot actual vs predicted
plt.figure(figsize=(12, 6))

plt.plot(test_dates, y_test.values, label="Actual Close")
plt.plot(test_dates, y_pred, label="Predicted Close")

plt.title("Actual vs Predicted Yes Bank Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()