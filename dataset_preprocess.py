import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
file_path = 'mock_training_dataset.csv'
data = pd.read_csv(file_path)

# Handle missing values
data.fillna(data.median(numeric_only=True), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical features
data = pd.get_dummies(data, columns=["Strand", "favorite_subject", "extracurricular_activities", "preferred_work_environment"])

# Label encode the target variable
label_encoder = LabelEncoder()
data["career_interest"] = label_encoder.fit_transform(data["career_interest"])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ["GWA", "Math_grade", "English_grade", "science_grade", "History_grade"]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Create additional features
data["overall_performance"] = data[["Math_grade", "English_grade", "science_grade", "History_grade"]].mean(axis=1)
socioeconomic_map = {"Low-income": 1, "Middle-income": 2, "High-income": 3}
data["socio-economic_status"] = data["socio-economic_status"].map(socioeconomic_map)
data["has_activities"] = data["extracurricular_activities"].apply(lambda x: 0 if x == "None" else 1)

# Visualize class imbalance
data["career_interest"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Career Interest Distribution")
plt.xlabel("Career Fields")
plt.ylabel("Number of Students")
plt.show()

# Save preprocessed data
data.to_csv('preprocessed_dataset.csv', index=False)
print("Preprocessed dataset saved as 'preprocessed_dataset.csv'.")
