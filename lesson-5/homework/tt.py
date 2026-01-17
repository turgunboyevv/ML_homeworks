# ==========================================
# Polynomial Features - Diabetes Dataset
# (90-point level solution)
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())

print("\nSummary statistics:")
print(X.describe())

# 2. Polynomial features (bmi, bp, s5)
poly_cols = ["bmi", "bp", "s5"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_part = poly.fit_transform(X[poly_cols])

poly_names = poly.get_feature_names_out(poly_cols)

print("\nPolynomial feature names:")
print(poly_names)

# Combine with remaining features
X_poly = pd.concat(
    [
        X.drop(columns=poly_cols).reset_index(drop=True),
        pd.DataFrame(X_poly_part, columns=poly_names)
    ],
    axis=1
)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_poly, X_test_poly, _, _ = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# 4. Train models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# 5. Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_poly = poly_model.predict(X_test_poly)

# 6. Evaluation
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

lin_mae, lin_rmse, lin_r2 = evaluate(y_test, y_pred_linear)
poly_mae, poly_rmse, poly_r2 = evaluate(y_test, y_pred_poly)

print("\nLinear Regression:")
print("MAE:", lin_mae)
print("RMSE:", lin_rmse)
print("R2:", lin_r2)

print("\nPolynomial Regression:")
print("MAE:", poly_mae)
print("RMSE:", poly_rmse)
print("R2:", poly_r2)

# 7. Top coefficients (Polynomial model)
coef_df = pd.DataFrame({
    "Feature": X_poly.columns,
    "Coefficient": poly_model.coef_
})

coef_df["Abs"] = coef_df["Coefficient"].abs()

print("\nTop 5 polynomial coefficients:")
print(coef_df.sort_values("Abs", ascending=False).head(5))

# 8. Visualization: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred_linear, label="Linear")
plt.scatter(y_test, y_pred_poly, label="Polynomial")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()

# 9. Simple prediction for given input
new_sample = pd.DataFrame([{
    "age": 0.05,
    "sex": 0.02,
    "bmi": 0.04,
    "bp": 0.03,
    "s1": -0.02,
    "s2": -0.01,
    "s3": 0.00,
    "s4": 0.02,
    "s5": 0.03,
    "s6": 0.01
}])

lin_pred = linear_model.predict(new_sample)

poly_part_new = poly.transform(new_sample[poly_cols])
new_poly = pd.concat(
    [
        new_sample.drop(columns=poly_cols).reset_index(drop=True),
        pd.DataFrame(poly_part_new, columns=poly_names)
    ],
    axis=1
)

poly_pred = poly_model.predict(new_poly)

print("\nPrediction for new sample:")
print("Linear model:", lin_pred[0])
print("Polynomial model:", poly_pred[0])

# 10. Short conclusion
print("\nConclusion:")
print("Polynomial features allow the model to capture non-linear patterns.")
print("However, improvement is limited due to small dataset size.")
