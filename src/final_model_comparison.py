import pandas as pd

# ==========================================
# MODEL RESULTS
# ==========================================

results = [
    {
        "Model": "Linear Regression",
        "Features": "Open, High, Low",
        "MAE": 9.7843,
        "MSE": 250.3307,
        "RMSE": 15.8218,
        "R2": 0.9848
    },
    {
        "Model": "Random Forest",
        "Features": "Open, High, Low",
        "MAE": 13.9084,
        "MSE": 461.2433,
        "RMSE": 21.4766,
        "R2": 0.9719
    },
    {
        "Model": "Linear Regression V2",
        "Features": "Open, High, Low, Previous_Close",
        "MAE": 10.4629,
        "MSE": 273.5210,
        "RMSE": 16.5385,
        "R2": 0.9833
    },
    {
        "Model": "Random Forest V2",
        "Features": "Open, High, Low, Previous_Close",
        "MAE": 14.1890,
        "MSE": 556.5005,
        "RMSE": 23.5903,
        "R2": 0.9661
    },
    {
        "Model": "Linear Regression V3",
        "Features": "Technical Features",
        "MAE": 9.8376,
        "MSE": 245.8189,
        "RMSE": 15.6786,
        "R2": 0.9850
    },
    {
        "Model": "Random Forest V3",
        "Features": "Technical Features",
        "MAE": 14.3017,
        "MSE": 529.1807,
        "RMSE": 23.0039,
        "R2": 0.9678
    }
]

# Create DataFrame
comparison = pd.DataFrame(results)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("========== FINAL MODEL COMPARISON ==========")

print(comparison.to_string(index=False))

# ==========================================
# SAVE RESULTS
# ==========================================

comparison.to_csv(
    "data/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

print("\nSaved file:")
print("data/model_comparison.csv")