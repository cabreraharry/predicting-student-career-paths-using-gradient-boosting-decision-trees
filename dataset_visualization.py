import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
unprocessed_file = "mock_training_dataset.csv"  # Unprocessed dataset
processed_file = "preprocessed_data.csv"  # Processed dataset

unprocessed_data = pd.read_csv(unprocessed_file)
processed_data = pd.read_csv(processed_file)

# Visualization 1: Class Distribution (Before and After Processing)
def plot_class_distribution(data, column_name, title, ax):
    data[column_name].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Count")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plot_class_distribution(unprocessed_data, "career_interest", "Unprocessed Data: Career Interest Distribution", axs[0])
plot_class_distribution(processed_data, "career_interest", "Processed Data: Career Interest Distribution", axs[1])
plt.tight_layout()
plt.show()

# Visualization 2: Feature Distributions (Numerical Features)
def plot_feature_distributions(data, columns, title):
    data[columns].hist(bins=20, figsize=(14, 10), color="skyblue")
    plt.suptitle(title, fontsize=16)
    plt.show()

numerical_cols = ["GWA", "Math_grade", "English_grade", "science_grade", "History_grade"]

# Unprocessed numerical feature distribution
plot_feature_distributions(unprocessed_data, numerical_cols, "Unprocessed Data: Numerical Feature Distributions")

# Processed numerical feature distribution
plot_feature_distributions(processed_data, numerical_cols, "Processed Data: Numerical Feature Distributions")

# Visualization 3: Correlation Heatmap (Processed Data)
plt.figure(figsize=(12, 8))
sns.heatmap(processed_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Processed Data: Feature Correlation Heatmap")
plt.show()

# Visualization 4: Pairplot for Key Features
def plot_pairplot(data, title, target_column="career_interest"):
    plt.figure(figsize=(12, 8))
    sns.pairplot(data=data, vars=numerical_cols, hue=target_column, palette="husl", diag_kind="kde")
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

# Pairplot of processed data
plot_pairplot(processed_data, "Processed Data: Pairplot of Key Features")
