import pandas as pd

# Model comparison results
comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Linear Regression V2",
        "Random Forest V2"
    ],
    "Features": [
        "Open, High, Low",
        "Open, High, Low",
        "Open, High, Low, Previous_Close",
        "Open, High, Low, Previous_Close"
    ],
    "MAE": [
        9.7843,
        13.9084,
        10.4629,
        14.1890
    ],
    "MSE": [
        250.3307,
        461.2433,
        273.5210,
        556.5005
    ],
    "RMSE": [
        15.8218,
        21.4766,
        16.5385,
        23.5903
    ],
    "R2": [
        0.9848,
        0.9719,
        0.9833,
        0.9661
    ]
})

print("========== FINAL MODEL COMPARISON ==========")
print(comparison.to_string(index=False))