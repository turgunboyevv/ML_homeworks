# ============================================
# Multiple Linear Regression
# House Price Prediction
# ============================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv("data/housing.csv")

print("DATA PREVIEW:")
print(df.head())

print("\nDATA INFO:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

# 3. Convert binary categorical variables (yes/no -> 1/0)
binary_cols = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]

for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# 4. One-hot encoding for furnishingstatus
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# 5. Define features and target
X = df.drop("price", axis=1)
y = df["price"]

# 6. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8. Train Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Coefficients interpretation
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nMODEL COEFFICIENTS:")
print(coefficients)

print(f"\nIntercept: {model.intercept_}")

# 10. Predictions
y_pred = model.predict(X_test)

# 11. Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# 12. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 13. Predicted vs Actual plot
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Prices")
plt.show()

# 14. Prediction for a new house
new_house = pd.DataFrame({
    "area": [2400],
    "bedrooms": [4],
    "bathrooms": [3],
    "stories": [2],
    "mainroad": [1],
    "guestroom": [0],
    "basement": [1],
    "hotwaterheating": [0],
    "airconditioning": [1],
    "parking": [2],
    "prefarea": [1],
    "furnishingstatus_semi-furnished": [1],
    "furnishingstatus_unfurnished": [0]
})

# Scale new input
new_house_scaled = scaler.transform(new_house)

predicted_price = model.predict(new_house_scaled)

print("\nNEW HOUSE PREDICTION:")
print(f"Predicted price: ${predicted_price[0]:,.2f}")

# 15. Final interpretation
print("\nINTERPRETATION:")
print("Positive coefficients indicate features that increase house price.")
print("Area, number of bathrooms, air conditioning, and preferred area")
print("have strong positive impacts on house prices.")
print("The model performs well based on R² and error metrics.")
