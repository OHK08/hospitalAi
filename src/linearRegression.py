import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
X_train = pd.read_csv("../data/cleaned/X_train.csv")
X_test = pd.read_csv("../data/cleaned/X_test.csv")
y_train = pd.read_csv("../data/cleaned/y_train.csv").squeeze()
y_test = pd.read_csv("../data/cleaned/y_test.csv").squeeze()

# Remove directly dependent features
drop_features = [
    "Time to Registration (min)",
    "Time to Triage (min)",
    "Time to Medical Professional (min)"
]

X_train = X_train.drop(columns=drop_features, errors="ignore")
X_test = X_test.drop(columns=drop_features, errors="ignore")

# Handle NaN values in y_train and y_test
y_train = y_train.fillna(y_train.mean())  # Replace NaN with mean
y_test = y_test.fillna(y_test.mean())    # Replace NaN with mean

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop Feature Importances (by absolute value):")
print(feature_importance.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance.head(15),
    x='Coefficient',
    y='Feature',
    hue='Feature',
    palette='viridis',
    dodge=False
)
plt.title("Top 15 Feature Importances (Linear Regression Coefficients)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Wait Time (min)")
plt.ylabel("Predicted Wait Time (min)")
plt.title("Actual vs. Predicted Wait Times")
plt.tight_layout()
plt.show()