import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Test dates
test_dates = df["Date"].iloc[len(X_train):]

# Plot actual vs predicted
plt.figure(figsize=(12, 6))

plt.plot(
    test_dates,
    y_test.values,
    label="Actual Close"
)

plt.plot(
    test_dates,
    y_pred,
    label="Random Forest Predicted Close"
)

plt.title("Random Forest: Actual vs Predicted Yes Bank Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()