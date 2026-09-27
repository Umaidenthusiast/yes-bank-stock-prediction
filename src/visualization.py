import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Select numerical columns
numerical_columns = ["Open", "High", "Low", "Close"]

# Calculate correlation matrix
correlation_matrix = df[numerical_columns].corr()

print("========== CORRELATION MATRIX ==========")
print(correlation_matrix.round(6))

# Plot correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Yes Bank Stock Prices")
plt.tight_layout()

plt.show()