import pandas as pd
import joblib


# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

model = joblib.load("models/linear_regression_v3.pkl")
scaler = joblib.load("models/scaler_v3.pkl")

print("Model loaded successfully!")
print("Scaler loaded successfully!")


# ==========================================
# REQUIRED FEATURES
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

print("\nRequired Features:")
for feature in features:
    print("-", feature)


# ==========================================
# USER INPUT
# ==========================================

print("\n========== ENTER STOCK DATA ==========")

while True:

    try:
        open_price = float(input("Enter Open price: "))
        high_price = float(input("Enter High price: "))
        low_price = float(input("Enter Low price: "))
        previous_close = float(input("Enter Previous Close price: "))

    except ValueError:
        print("Invalid input: Please enter numeric values only.\n")
        continue

    # High cannot be lower than Open
    if high_price < open_price:
        print("Invalid input: High price cannot be lower than Open price.")
        print("Please enter the stock data again.\n")
        continue

    # Low cannot be higher than Open
    if low_price > open_price:
        print("Invalid input: Low price cannot be higher than Open price.")
        print("Please enter the stock data again.\n")
        continue

    # Low cannot be higher than High
    if low_price > high_price:
        print("Invalid input: Low price cannot be higher than High price.")
        print("Please enter the stock data again.\n")
        continue

    # Previous close must be positive
    if previous_close <= 0:
        print("Invalid input: Previous Close must be greater than 0.")
        print("Please enter the stock data again.\n")
        continue

    break


# ==========================================
# CALCULATE TECHNICAL FEATURES
# ==========================================

return_value = (
    open_price - previous_close
) / previous_close

ma_3 = (
    previous_close + open_price + high_price
) / 3

volatility_3 = (
    high_price - low_price
) / open_price


# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame(
    [[
        open_price,
        high_price,
        low_price,
        previous_close,
        return_value,
        ma_3,
        volatility_3
    ]],
    columns=features
)

print("\n========== INPUT DATA ==========")
print(input_data)


# ==========================================
# SCALE INPUT
# ==========================================

scaled_input = scaler.transform(input_data)

print("\n========== SCALED INPUT ==========")
print(scaled_input)


# ==========================================
# MAKE PREDICTION
# ==========================================

prediction = model.predict(scaled_input)

predicted_close = prediction[0]

print("\n========== PREDICTION ==========")
print(f"Predicted Close Price: {predicted_close:.2f}")