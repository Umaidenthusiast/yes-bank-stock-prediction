import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Create scaler
scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the training scaler
X_test_scaled = scaler.transform(X_test)

print("========== SCALING ==========")

print("Training data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

print("\n========== FIRST 5 SCALED TRAINING ROWS ==========")
print(X_train_scaled[:5])

print("\n========== FIRST 5 SCALED TESTING ROWS ==========")
print(X_test_scaled[:5])

print("\n========== TRAINING SCALED MEAN ==========")
print(X_train_scaled.mean(axis=0))

print("\n========== TRAINING SCALED STD ==========")
print(X_train_scaled.std(axis=0))