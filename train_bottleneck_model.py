# === Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score
)
from xgboost import XGBClassifier

# === 1. Load CSV ===
df = pd.read_csv("final_files/ErWaitTime.csv")

# === 2. Convert Visit Date to features ===
df["Visit Date"] = pd.to_datetime(df["Visit Date"])
df["Visit_Month"] = df["Visit Date"].dt.month
df["Visit_Hour"] = df["Visit Date"].dt.hour
df["Is_Weekend"] = df["Visit Date"].dt.dayofweek.isin([5, 6]).astype(int)
df["Day of Week"] = df["Visit Date"].dt.day_name()

# Season + TOD helpers
def get_season(m):
    if m in [12,1,2]: return "Winter"
    if m in [3,4,5]: return "Spring"
    if m in [6,7,8]: return "Summer"
    return "Fall"

def get_time_of_day(h):
    if 5<=h<12: return "Morning"
    if 12<=h<17: return "Afternoon"
    if 17<=h<21: return "Evening"
    return "Night"

df["Season"] = df["Visit_Month"].apply(get_season)
df["Time of Day"] = df["Visit_Hour"].apply(get_time_of_day)

# === 3. Simplify hospital to cluster ===
df["Hospital_Cluster"] = df["Region"].map({
    "Urban":"Urban_Hospital",
    "Rural":"Rural_Hospital",
    "Semi-Urban":"Urban_Hospital"
})

# === 4. Create Bottleneck_Stage ===
stage_cols = ['Time to Registration (min)',
              'Time to Triage (min)',
              'Time to Medical Professional (min)']
df['Bottleneck_Stage'] = df[stage_cols].idxmax(axis=1)
df['Bottleneck_Stage'] = df['Bottleneck_Stage'].replace({
    'Time to Registration (min)': 'Registration',
    'Time to Triage (min)': 'Triage',
    'Time to Medical Professional (min)': 'Medical'
})

# === 5. Use wait time as model1-predicted feature ===
df['Predicted_Wait_Time'] = df['Total Wait Time (min)']  # during inference your Flask code will use model1.predict()

# === 6. Select only the final desired features ===
final_features = [
    "Region", "Hospital_Cluster", "Visit_Hour", "Visit_Month", "Is_Weekend",
    "Season", "Time of Day", "Urgency Level", "Nurse-to-Patient Ratio",
    "Specialist Availability", "Facility Size (Beds)", "Predicted_Wait_Time"
]

X = df[final_features]
y = df['Bottleneck_Stage']

# === 7. Label encode target ===
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))

# === 8. Column types ===
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = X.select_dtypes(exclude=['object']).columns.tolist()

# === 9. Pipeline ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    n_estimators=350,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9
)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('model', model)])

# === 10. Split and Train ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)

# === 11. Evaluation ===
y_pred = clf.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap=plt.cm.Blues)
plt.show()

# === 12. Save ===
joblib.dump(clf, "./hospitalAi/bottleneck_classifier_xgb.pkl")
joblib.dump(le,  "./hospitalAi/label_encoder.pkl")
print("Model + LabelEncoder saved")




# | Metric                 | **First Model**                                                                | **Second Model**                   | **Analysis**                                                       |
# | ---------------------- | ------------------------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------------ |
# | **Accuracy**           | 0.98                                                                           | 0.98                               | Same overall accuracy                                              |
# | **Balanced Accuracy**  | 0.8436                                                                         | 0.8179                             | First model slightly better at handling class imbalance            |
# | **Macro F1-Score**     | 0.8759                                                                         | 0.8574                             | First model has slightly higher F1 across classes                  |
# | **Triage Recall**      | 0.69                                                                           | 0.64                               | First model detects minority class (Triage) better                 |
# | **Triage Precision**   | 0.84                                                                           | 0.83                               | Very similar, slightly better in first model                       |
# | **Feature Importance** | Time to Medical Professional, Time to Triage, Total Wait Time are top features | Urgency Level dominates importance | First model’s top features align more intuitively with bottlenecks |
# First Model is better

# Second Model
# === Import Libraries ===
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     balanced_accuracy_score,
#     f1_score
# )
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline
#
# # === 1. Load CSV ===
# df = pd.read_csv("./ErWaitTime.csv")
#
# # === 2. Drop Low-Value Identifier Columns ===
# df.drop(columns=["Visit ID", "Patient ID", "Hospital Name", "Hospital ID"], inplace=True)
#
# # === 3. Convert Visit Date to numeric features ===
# df["Visit Date"] = pd.to_datetime(df["Visit Date"])
# df["Visit_Month"] = df["Visit Date"].dt.month
# df["Visit_Hour"] = df["Visit Date"].dt.hour
# df["Is_Weekend"] = df["Visit Date"].dt.dayofweek.isin([5, 6]).astype(int)
# df.drop(columns=["Visit Date"], inplace=True)
#
# # === 4. Simplify Hospital into Urban/Rural clusters ===
# df["Hospital_Cluster"] = df["Region"].map(
#     {"Urban": "Urban_Hospital", "Rural": "Rural_Hospital", "Semi-Urban": "Urban_Hospital"}
# )
#
# # === 5. Create Bottleneck Stage Column ===
# stage_cols = ['Time to Registration (min)', 'Time to Triage (min)', 'Time to Medical Professional (min)']
# df['Bottleneck_Stage'] = df[stage_cols].idxmax(axis=1)
#
# rename_map = {
#     'Time to Registration (min)': 'Registration',
#     'Time to Triage (min)': 'Triage',
#     'Time to Medical Professional (min)': 'Medical'
# }
# df['Bottleneck_Stage'] = df['Bottleneck_Stage'].replace(rename_map)
#
# # === 6. Define X (features) and y (label) ===
# X = df.drop(columns=['Bottleneck_Stage'])
# y = df['Bottleneck_Stage']
#
# # === 7. Encode labels numerically ===
# le = LabelEncoder()
# y = le.fit_transform(y)
# num_classes = len(np.unique(y))
#
# # === 8. Identify categorical vs. numerical columns ===
# categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
# numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
#
# # === 9. Preprocessing (OneHotEncode categoricals) ===
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#         ('num', 'passthrough', numerical_cols)
#     ]
# )
#
# # === 10. Train-test Split ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# # === 11. Build imbalanced-aware pipeline ===
# model = XGBClassifier(
#     objective='multi:softmax',
#     num_class=num_classes,
#     eval_metric='mlogloss',
#     n_estimators=350,
#     learning_rate=0.08,
#     max_depth=6,
#     subsample=0.9,
#     colsample_bytree=0.9
# )
#
# imb_pipeline = ImbPipeline(steps=[
#     ('preprocessor', preprocessor),   # OneHotEncoding
#     ('smote', SMOTE(random_state=42)),  # Handles numeric output from preprocessor
#     ('model', model)
# ])
#
# # === 12. Fit pipeline on training set ===
# imb_pipeline.fit(X_train, y_train)
#
# # === 13. Evaluation ===
# y_pred = imb_pipeline.predict(X_test)
#
# print("=== Classification Report ===")
# print(classification_report(y_test, y_pred, target_names=le.classes_))
#
# bal_acc = balanced_accuracy_score(y_test, y_pred)
# print(f"\nBalanced Accuracy: {bal_acc:.4f}")
#
# macro_f1 = f1_score(y_test, y_pred, average='macro')
# print(f"Macro F1-Score: {macro_f1:.4f}")
#
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()
#
# # === 14. Feature Importance Table ===
# xgb_model = imb_pipeline.named_steps['model']
# ohe = imb_pipeline.named_steps['preprocessor'].named_transformers_['cat']
# encoded_cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
# full_feature_list = encoded_cat_features + numerical_cols
#
# importances = xgb_model.feature_importances_
#
# feature_importance_table = pd.DataFrame({
#     'Feature': full_feature_list,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)
#
# print("\n=== Top 25 Most Important Features ===")
# print(feature_importance_table.head(25))
#
# # === 15. Save Model and Label Encoder ===
# joblib.dump(imb_pipeline, "bottleneck_classifier_xgb.pkl")
# joblib.dump(le, "label_encoder.pkl")
# print("\nModel saved as bottleneck_classifier_xgb.pkl")
# print("Label Encoder saved as label_encoder.pkl")
