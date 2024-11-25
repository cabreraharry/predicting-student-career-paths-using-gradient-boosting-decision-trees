# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed dataset
file_path = "preprocessed_dataset.csv"  # Adjust path if necessary
data = pd.read_csv(file_path)

# Step 1: Split Data into Features (X) and Target (y)
X = data.drop(columns=["career_interest", "students_id"])  # Features
y = data["career_interest"]  # Target

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train the XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Save the Model for Future Use
import joblib
joblib.dump(model, "trained_xgboost_model.pkl")
print("Model saved as 'trained_xgboost_model.pkl'.")
