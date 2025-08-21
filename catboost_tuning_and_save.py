import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------------
# 1. Preprocessing (same as before)
# ---------------------------------
df = pd.read_csv("final_files/ErWaitTime.csv")
df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce")
df.dropna(subset=["Visit Date"], inplace=True)

df["Day of Week"] = df["Visit Date"].dt.day_name()
df["Hour"] = df["Visit Date"].dt.hour

def get_season(m):
    return "Winter" if m in [12,1,2] else "Spring" if m in [3,4,5] else "Summer" if m in [6,7,8] else "Fall"

def get_time_of_day(h):
    if 5 <= h < 12: return "Morning"
    if 12 <= h < 17: return "Afternoon"
    if 17 <= h < 21: return "Evening"
    return "Night"

df["Season"] = df["Visit Date"].dt.month.apply(get_season)
df["Time of Day"] = df["Hour"].apply(get_time_of_day)

features = [
    "Region","Day of Week","Season","Time of Day","Urgency Level",
    "Nurse-to-Patient Ratio","Specialist Availability","Facility Size (Beds)"
]
target = "Total Wait Time (min)"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns by index
cat_features = np.where(X_train.dtypes == object)[0]

# ---------------------------------
# 2. Wrapper for CatBoost -> sklearn
# ---------------------------------
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor

class CatBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, depth=6, iterations=700, learning_rate=0.1, l2_leaf_reg=3, cat_features=None):
        self.depth = depth
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.cat_features = cat_features
        self.model = None

    def fit(self, X, y):
        self.model = CatBoostRegressor(
            depth=self.depth,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function='RMSE',
            verbose=0
        )
        self.model.fit(X, y, cat_features=self.cat_features)
        return self

    def predict(self, X):
        return self.model.predict(X)

# ---------------------------------
# 3. Tune using sklearn GridSearch
# ---------------------------------
wrapper = CatBoostWrapper(cat_features=cat_features)

param_grid = {
    'depth': [4, 6, 8],
    'iterations': [500, 800],
    'learning_rate': [0.03, 0.1],
    'l2_leaf_reg': [3, 9]
}

grid = GridSearchCV(
    estimator=wrapper,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2
)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)

# ---------------------------------
# 4. Evaluate
# ---------------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

# ---------------------------------
# 5. Save model
# ---------------------------------
joblib.dump(best_model, "catboost_best.pkl")
print("Model saved as catboost_best.pkl")

# MAE : 17.16
# RMSE: 25.65
# R2  : 0.8586
