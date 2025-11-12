import pandas as pd

# Load dataset
data = pd.read_csv("/Users/RoseKamau/Downloads/creditcard.csv")

print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)
print("\nColumns:\n", data.columns.tolist())
print("\nPreview:\n", data.head())

# --- Step 2: Data Exploration ---
print("\n✅ Data Exploration\n")

# Check for missing values
print("Missing values per column:\n", data.isnull().sum().sort_values(ascending=False).head())

# Check class balance (fraud vs non-fraud)
print("\nClass distribution:\n", data['Class'].value_counts())

# Calculate the percentage of fraud transactions
fraud_percentage = (data['Class'].value_counts()[1] / len(data)) * 100
print(f"\nFraudulent transactions: {fraud_percentage:.4f}%")

# Display some basic statistics
print("\nSummary statistics for numerical features:\n", data.describe())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("✅ Preprocessing complete!")
print("Original training samples:", len(y_train))
print("After SMOTE:", len(y_train_res))
print("Fraud samples after resampling:", sum(y_train_res))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# STEP 3: MODEL TRAINING
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# -------------------------------
# STEP 4: EVALUATION
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("✅ Model training complete!")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from imblearn.over_sampling import SMOTE

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", X_train.shape, "After SMOTE:", X_train_resampled.shape)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Credit Card Fraud Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
importances = model.feature_importances_
indices = importances.argsort()[-10:][::-1]  # top 10 features
plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.title("Top 10 Most Important Features in Fraud Detection")
plt.xlabel("Feature Importance Score")
plt.show()