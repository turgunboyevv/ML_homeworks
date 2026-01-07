import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
data = pd.read_csv("housing.csv")

# 3. Quick exploration
print(data.head())
print(data.info())
print(data.describe())

# Check missing values
print(data.isnull().sum())

# Scatter plot: area vs price
plt.figure(figsize=(7,5))
sns.scatterplot(x="area", y="price", data=data)
plt.title("House Area vs Price")
plt.xlabel("Area (sqft)")
plt.ylabel("Price ($)")
plt.show()

# 4. Train-test split
X = data[["area"]]   # independent variable
y = data["price"]    # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Linear Regression using scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)

# Get slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Plot regression line
plt.figure(figsize=(7,5))
sns.scatterplot(x="area", y="price", data=data, label="Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Area (sqft)")
plt.ylabel("Price ($)")
plt.title("Linear Regression: Area vs Price")
plt.legend()
plt.show()

# 9. Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(7,5))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Error")
plt.show()

# 10. Prediction for a 1000 sqft house
pred_price = model.predict([[1000]])
print(f"Predicted price for 1000 sqft house: ${pred_price[0]:.2f}")
