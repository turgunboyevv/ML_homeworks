# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


# ===============================
# 2. LOAD DATA
# ===============================
df = pd.read_csv("data/car_price.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


# ===============================
# 3. HANDLE MISSING VALUES
# ===============================
print("\nMissing values:")
print(df.isnull().sum())

df = df.fillna(df.median(numeric_only=True))


# ===============================
# 4. FEATURE / TARGET SPLIT
# ===============================
y = df["price"]
X = df.drop(columns=["price"])


# ===============================
# 5. FEATURE SCALING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# 6. TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ===============================
# 7. COST FUNCTION
# ===============================
def compute_cost(X, y, weights, bias):
    m = len(y)
    y_pred = X.dot(weights) + bias
    return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)


# ===============================
# 8. GRADIENT DESCENT
# ===============================
def gradient_descent(X, y, lr=0.05, iterations=1500):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    cost_history = []

    for i in range(iterations):
        y_pred = X.dot(weights) + bias

        dw = (1 / m) * X.T.dot(y_pred - y)
        db = (1 / m) * np.sum(y_pred - y)

        weights -= lr * dw
        bias -= lr * db

        cost = compute_cost(X, y, weights, bias)
        cost_history.append(cost)

        if i % 200 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return weights, bias, cost_history


# ===============================
# 9. TRAIN MODEL
# ===============================
weights, bias, cost_history = gradient_descent(
    X_train, y_train
)


# ===============================
# 10. COST FUNCTION PLOT
# ===============================
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Convergence")
plt.show()


# ===============================
# 11. EVALUATION (TEST SET)
# ===============================
y_test_pred = X_test.dot(weights) + bias

mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\nGradient Descent Results:")
print("MSE:", mse)
print("R2 Score:", r2)


# ===============================
# 12. VISUALIZATION (Horsepower vs Price)
# ===============================
if "horsepower" in df.columns:
    hp_index = X.columns.get_loc("horsepower")

    plt.figure()
    plt.scatter(df["horsepower"], y, label="Actual")
    plt.scatter(
        df["horsepower"],
        X_scaled.dot(weights) + bias,
        color="red",
        label="Predicted"
    )
    plt.xlabel("Horsepower")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# ===============================
# 13. SKLEARN COMPARISON (BONUS)
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_sk_pred = lr_model.predict(X_test)

print("\nScikit-learn Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_sk_pred))
print("R2:", r2_score(y_test, y_sk_pred))
