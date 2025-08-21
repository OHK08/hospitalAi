# hyper_tuning_and_save_log.py
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from final_files.preprocessing import load_and_preprocess
from final_files.base_model import build_base_pipe

# 1. Load + preprocess using the existing utility
X, y, features = load_and_preprocess("final_files/ErWaitTime.csv")

# 2. Log-transform the target
y_log = np.log1p(y)

# 3. Build your base ML pipeline (from base_model.py)
pipe = build_base_pipe(features)

# 4. Hyperparameter grid
param_grid = {
    'regressor__n_estimators':      [200, 400],
    'regressor__max_depth':        [3, 5, 7],
    'regressor__learning_rate':    [0.05, 0.1],
    'regressor__subsample':        [0.7, 0.9],
    'regressor__colsample_bytree': [0.7, 0.9, 1.0],
}

# 5. Perform Grid Search
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid.fit(X, y_log)
print("BEST PARAMS:", grid.best_params_)

# 6. Evaluate best model
best_model = grid.best_estimator_
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train_log)

y_pred_log = best_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)      # inverse transform for evaluation
y_true     = np.expm1(y_test_log)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

# 7. Save tuned model
joblib.dump(best_model, "log_transformed_xgb.pkl")
print("Model saved as log_transformed_xgb.pkl")

# MAE: 16.83
# RMSE: 25.83
# R2: 0.8566
