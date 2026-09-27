import pandas as pd

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

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

print("========== MODEL TRAINING ==========")
print("Model:", model)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\n========== MODEL COEFFICIENTS ==========")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("\nIntercept:", round(model.intercept_, 4))

print("\nModel training completed successfully!")