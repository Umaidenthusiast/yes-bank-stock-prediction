import pandas as pd

print("==============================================")
print("       YES BANK STOCK PRICE PREDICTION")
print("              FINAL MODEL REPORT")
print("==============================================")

# ==========================================
# MODEL COMPARISON
# ==========================================

comparison_file = "data/model_comparison.csv"

df = pd.read_csv(comparison_file)

print("\n========== MODEL COMPARISON ==========")
print(df.to_string(index=False))

# ==========================================
# BEST LINEAR REGRESSION MODEL RESULTS
# ==========================================

print("\n========== LINEAR REGRESSION V3 ==========")

print("Features:")
features = [
    "Open",
    "High",
    "Low",
    "Previous_Close",
    "Return",
    "MA_3",
    "Volatility_3"
]

for feature in features:
    print("-", feature)

print("\nPerformance:")
v3 = df[df["Model"] == "Linear Regression V3"]

if not v3.empty:
    print("MAE :", v3.iloc[0]["MAE"])
    print("MSE :", v3.iloc[0]["MSE"])
    print("RMSE:", v3.iloc[0]["RMSE"])
    print("R²  :", v3.iloc[0]["R2"])

# ==========================================
# RESIDUAL ANALYSIS
# ==========================================

print("\n========== RESIDUAL ANALYSIS ==========")

print("Mean Residual      : -5.0423")
print("Minimum Residual   : -62.9409")
print("Maximum Residual   : 23.3998")
print("Residual Std Dev   : 14.8457")

# ==========================================
# VIF ANALYSIS
# ==========================================

print("\n========== VIF ANALYSIS ==========")

vif_data = {
    "Feature": ["Open", "High", "Low"],
    "VIF": [81.926279, 79.199072, 34.563462]
}

vif_df = pd.DataFrame(vif_data)

print(vif_df.to_string(index=False))

# ==========================================
# MODEL FILES
# ==========================================

print("\n========== SAVED MODEL ==========")

print("Model : models/linear_regression_v3.pkl")
print("Scaler: models/scaler_v3.pkl")

print("\n==============================================")
print("             REPORT COMPLETED")
print("==============================================")