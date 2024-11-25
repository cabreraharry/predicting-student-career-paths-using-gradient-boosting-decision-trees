import pandas as pd
import random

# Generate a realistic mock dataset for training and testing
random.seed(42)

# Number of samples
n_samples = 1000

# Generate mock data
data = {
    "students_id": range(1, n_samples + 1),
    "age": [random.randint(16, 19) for _ in range(n_samples)],
    "gender": [random.choice(["Male", "Female"]) for _ in range(n_samples)],
    "Strand": [random.choice(["STEM", "ABM", "HUMSS", "GAS", "TVL"]) for _ in range(n_samples)],
    "GWA": [round(random.uniform(75, 95), 2) for _ in range(n_samples)],
    "Math_grade": [round(random.uniform(75, 95), 2) for _ in range(n_samples)],
    "English_grade": [round(random.uniform(75, 95), 2) for _ in range(n_samples)],
    "science_grade": [round(random.uniform(75, 95), 2) for _ in range(n_samples)],
    "History_grade": [round(random.uniform(75, 95), 2) for _ in range(n_samples)],
    "favorite_subject": [random.choice(["Math", "Science", "English", "Arts", "History"]) for _ in range(n_samples)],
    "career_interest": [random.choice([
        "Engineering", "Arts", "Business", "Education", 
        "Science and Technology", "Farming and Forestry", "Health and Medicine"
    ]) for _ in range(n_samples)],
    "extracurricular_activities": [random.choice(["Debate", "Sports", "Music", "NSTP", "None"]) for _ in range(n_samples)],
    "socio-economic_status": [random.choice(["Low-income", "Middle-income", "High-income"]) for _ in range(n_samples)],
    "preferred_work_environment": [random.choice(["Office", "Field", "Remote"]) for _ in range(n_samples)],
}

# Convert to DataFrame
mock_dataset = pd.DataFrame(data)

# Save to a CSV file for use
file_path = 'mock_training_dataset.csv'
mock_dataset.to_csv(file_path, index=False)

file_path
