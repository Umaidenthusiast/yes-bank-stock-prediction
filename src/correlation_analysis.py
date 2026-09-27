import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/yes_bank_stock.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# ==========================================
# CREATE TECHNICAL FEATURES
# ==========================================

df["Previous_Close"] = df["Close"].shift(1)

df["Return"] = df["Close"].pct_change()

df["MA_3"] = df["Close"].rolling(window=3).mean()

df["Volatility_3"] = df["Return"].rolling(window=3).std()

# Remove missing values
df = df.dropna().reset_index(drop=True)

# ==========================================
# SELECT FEATURES
# ==========================================

features = [
    "Open",
    "High",
    "Low",
    "Previous_Close",
    "Return",
    "MA_3",
    "Volatility_3",
    "Close"
]

# Calculate correlation matrix
correlation_matrix = df[features].corr()

print("========== CORRELATION MATRIX ==========")
print(correlation_matrix.round(4))

# ==========================================
# CORRELATION WITH CLOSE PRICE
# ==========================================

print("\n========== CORRELATION WITH CLOSE PRICE ==========")

close_correlation = correlation_matrix["Close"].sort_values(
    ascending=False
)

print(close_correlation.round(4))

# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10, 7))

plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Correlation Matrix - YES Bank Stock Features")

plt.tight_layout()

plt.show()