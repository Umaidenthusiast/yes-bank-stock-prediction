import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Select predictor variables
features = ["Open", "High", "Low"]

X = df[features]

# Calculate VIF
vif_data = pd.DataFrame()

vif_data["Feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("========== VIF ANALYSIS ==========")
print(vif_data)