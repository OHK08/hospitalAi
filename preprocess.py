import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Load the dataset
df = pd.read_csv('ErWaitTime.csv')

# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df['Nurse-to-Patient Ratio'] = df['Nurse-to-Patient Ratio'].fillna(df['Nurse-to-Patient Ratio'].median())
df['Specialist Availability'] = df['Specialist Availability'].fillna(df['Specialist Availability'].median())
df['Patient Satisfaction'] = df['Patient Satisfaction'].fillna(df['Patient Satisfaction'].mode()[0])

# Fill time columns
time_columns = ['Time to Registration (min)', 'Time to Triage (min)', 'Time to Medical Professional (min)']
for col in time_columns:
    df[col] = df[col].fillna(df[col].median())

# Parse Visit Date
df['Visit Date'] = pd.to_datetime(df['Visit Date'], errors='coerce')
df.dropna(subset=['Visit Date'], inplace=True)

# Drop unnecessary identifier columns
df.drop(columns=['Visit ID', 'Patient ID', 'Hospital ID', 'Visit Date'], inplace=True)

# Target variable
y = df['Total Wait Time (min)']
X = df.drop(columns=['Total Wait Time (min)'])

# Categorical columns
ordinal_cols = ['Hospital Name']
onehot_cols = ['Region', 'Day of Week', 'Season', 'Time of Day', 'Urgency Level', 'Patient Outcome']

# Ordinal encoding
ord_encoder = OrdinalEncoder()
X[ordinal_cols] = ord_encoder.fit_transform(X[ordinal_cols])

# One-hot encoding
X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)

# Scale numerical features
numerical_cols = ['Nurse-to-Patient Ratio', 'Specialist Availability', 'Facility Size (Beds)',
                  'Time to Registration (min)', 'Time to Triage (min)',
                  'Time to Medical Professional (min)', 'Patient Satisfaction']

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Final shape check
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save for ML training
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
