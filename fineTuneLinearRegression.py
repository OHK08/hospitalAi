import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# Remove directly dependent features
drop_features = [
    "Time to Registration (min)",
    "Time to Triage (min)",
    "Time to Medical Professional (min)"
]
X_train = X_train.drop(columns=drop_features, errors="ignore")
X_test = X_test.drop(columns=drop_features, errors="ignore")

# Handle NaN values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
y_train = y_train.fillna(y_train.mean())
y_test = y_test.fillna(y_test.mean())

# Remove outliers from training data
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
outliers = (y_train < (Q1 - 1.5 * IQR)) | (y_train > (Q3 + 1.5 * IQR))
X_train_clean = X_train[~outliers]
y_train_clean = y_train[~outliers]
print(f"Removed {outliers.sum()} outliers")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train_clean.index)

# Check for multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_clean.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled_df.values, i) for i in range(X_train_scaled_df.shape[1])]
print("\nVariance Inflation Factors (VIF):")
print(vif_data.sort_values("VIF", ascending=False))

# Remove features with high VIF (> 5 or 10, depending on tolerance)
high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
if high_vif_features:
    print(f"Removing high VIF features: {high_vif_features}")
    X_train_clean = X_train_clean.drop(columns=high_vif_features, errors="ignore")
    X_test = X_test.drop(columns=high_vif_features, errors="ignore")
    # Re-scale after dropping features
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test)

# Feature Selection with RFE
lr_model = LinearRegression()
n_features = min(10, X_train_clean.shape[1])  # Select top 10 or fewer if fewer features remain
rfe = RFE(lr_model, n_features_to_select=n_features)
rfe = rfe.fit(X_train_scaled, y_train_clean)
selected_features = X_train_clean.columns[rfe.support_].tolist()
print("\nSelected Features by RFE:", selected_features)

# Train model with selected features
X_train_rfe = X_train_scaled[:, rfe.support_]
X_test_rfe = X_test_scaled[:, rfe.support_]
lr_model.fit(X_train_rfe, y_train_clean)
y_pred = lr_model.predict(X_test_rfe)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nFine-Tuned Linear Regression Performance Metrics:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print("Baseline Metrics for Comparison: MAE: 17.07, RMSE: 24.08, R²: 0.8754")

# Feature Importance (for selected features)
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)
print("\nTop Feature Importances (Fine-Tuned Model):")
print(feature_importance.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance,
    x='Coefficient',
    y='Feature',
    hue='Feature',
    palette='viridis',
    dodge=False
)
plt.title("Top Feature Importances (Fine-Tuned Linear Regression)")
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Wait Time (min)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot (Fine-Tuned Model)")
plt.tight_layout()
plt.show()