# ===============================
# Simple Linear Regression
# House Price Prediction
# ===============================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv("data/housing.csv")

# 3. Basic exploration
print("DATA INFO:")
print(df.info())

print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())

print("\nMISSING VALUES:")
print(df.isnull().sum())

# 4. Scatter plot: Area vs Price
plt.figure()
plt.scatter(df["area"], df["price"])
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Size vs Price")
plt.show()

# 5. Prepare features and target
X = df[["area"]]
y = df["price"]

# 6. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Manual Linear Regression calculation
x = X_train["area"].values
y_manual = y_train.values

x_mean = np.mean(x)
y_mean = np.mean(y_manual)

m_manual = np.sum((x - x_mean) * (y_manual - y_mean)) / np.sum((x - x_mean)**2)
b_manual = y_mean - m_manual * x_mean

print(f"\nMANUAL LINEAR REGRESSION:")
print(f"Slope (m): {m_manual}")
print(f"Intercept (b): {b_manual}")

# 8. Scikit-learn Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

print("\nSCIKIT-LEARN MODEL:")
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

# 9. Predictions on test set
y_pred = model.predict(X_test)

# 10. Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# 11. Regression line visualization
plt.figure()
plt.scatter(X, y)
plt.plot(X, model.predict(X), linewidth=2)
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("Linear Regression Line")
plt.show()

# 12. Residual plot
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 13. Prediction for 1000 sq ft house
price_1000 = model.predict([[1000]])

print("\nPREDICTION:")
print(f"Predicted price for 1000 sq ft house: ${price_1000[0]:,.2f}")

# 14. Final interpretation
print("\nINTERPRETATION:")
print("There is a positive linear relationship between house size and price.")
print("As the area increases, the house price increases proportionally.")
print("The model performs well based on evaluation metrics.")
