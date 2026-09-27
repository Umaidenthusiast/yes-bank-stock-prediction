import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

model = joblib.load("models/linear_regression_v3.pkl")
scaler = joblib.load("models/scaler_v3.pkl")

print("Model loaded successfully!")
print("Scaler loaded successfully!")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/yes_bank_stock.csv")

df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

df = df.sort_values("Date").reset_index(drop=True)

# ==========================================
# CREATE TECHNICAL FEATURES
# ==========================================

df["Previous_Close"] = df["Close"].shift(1)

df["Return"] = df["Close"].pct_change()

df["MA_3"] = df["Close"].rolling(window=3).mean()

df["Volatility_3"] = df["Return"].rolling(window=3).std()

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

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

train_size = int(len(df) * 0.80)

test_df = df.iloc[train_size:].copy()

X_test = test_df[features]

y_test = test_df["Close"]

# ==========================================
# SCALE TEST DATA
# ==========================================

X_test_scaled = scaler.transform(X_test)

# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(X_test_scaled)

# ==========================================
# CREATE RESULT DATAFRAME
# ==========================================

results = pd.DataFrame({
    "Date": test_df["Date"],
    "Actual_Close": y_test.values,
    "Predicted_Close": predictions
})

print("\n========== FINAL PREDICTIONS ==========")
print(results.head(10))

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual_Close"],
    label="Actual Close Price"
)

plt.plot(
    results["Date"],
    results["Predicted_Close"],
    label="Predicted Close Price"
)

plt.title("YES Bank Stock Price: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.legend()
plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)

plt.tight_layout()

# ==========================================
# SAVE FIGURE
# ==========================================

plt.savefig(
    "data/final_prediction_plot.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nFinal prediction plot saved successfully!")
print("Saved file: data/final_prediction_plot.png")

plt.show()