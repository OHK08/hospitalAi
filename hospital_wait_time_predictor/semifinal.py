import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("../ErWaitTime.csv")

# ------------------------------
# 2. Date preprocessing (safe for mixed formats)
# ------------------------------
df["Visit Date"] = pd.to_datetime(df["Visit Date"], dayfirst=False, errors="coerce")

# Drop rows with invalid/missing dates
df = df.dropna(subset=["Visit Date"])

# Extract features from date
df["Day of Week"] = df["Visit Date"].dt.day_name()
df["Hour"] = df["Visit Date"].dt.hour

# Season mapping
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df["Season"] = df["Visit Date"].dt.month.apply(get_season)

# Time of day mapping
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["Time of Day"] = df["Hour"].apply(get_time_of_day)

# ------------------------------
# 3. Select features & target
# ------------------------------
features = [
    "Hospital Name",
    "Urgency Level",
    "Day of Week",
    "Season",
    "Time of Day",
    "Nurse-to-Patient Ratio",
    "Facility Size (Beds)"
]
target = "Total Wait Time (min)"

X = df[features]
y = df[target]

# ------------------------------
# 4. Preprocessing pipeline
# ------------------------------
categorical_cols = ["Hospital Name", "Urgency Level", "Day of Week", "Season", "Time of Day"]
numeric_cols = ["Nurse-to-Patient Ratio", "Facility Size (Beds)"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ------------------------------
# 5. Model pipeline
# ------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, random_state=42))
])

# ------------------------------
# 6. Train-test split & fit
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# ------------------------------
# 7. Save the trained pipeline as .pkl
# ------------------------------
joblib.dump(model, "../hospital_wait_time_predictor/model.pkl")
print("Model saved as model.pkl")

# ------------------------------
# 8. Example: Predict from user input
# ------------------------------
def predict_wait_time(hospital_name, urgency, date_str, nurse_ratio, facility_size):
    date_obj = pd.to_datetime(date_str, dayfirst=False, errors="coerce")
    input_df = pd.DataFrame([{
        "Hospital Name": hospital_name,
        "Urgency Level": urgency,
        "Day of Week": date_obj.day_name(),
        "Season": get_season(date_obj.month),
        "Time of Day": get_time_of_day(date_obj.hour),
        "Nurse-to-Patient Ratio": nurse_ratio,
        "Facility Size (Beds)": facility_size
    }])
    return model.predict(input_df)[0]

# Example usage
predicted_time = predict_wait_time(
    "Springfield General Hospital", "Medium", "2024-08-15 14:30:00", 4, 100
)
print(f"Predicted Wait Time: {predicted_time:.2f} minutes")
