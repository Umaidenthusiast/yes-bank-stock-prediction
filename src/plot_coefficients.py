import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/linear_regression_v3.pkl")

# Feature names used during training
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
# CREATE COEFFICIENT DATAFRAME
# ==========================================

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("========== LINEAR REGRESSION V3 COEFFICIENTS ==========")
print(coefficients)

# ==========================================
# PLOT COEFFICIENTS
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Linear Regression V3 - Feature Coefficients")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()