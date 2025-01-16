import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from dtreeviz.trees import dtreeviz  # For tree visualization

# Set a random seed for reproducibility
np.random.seed(42)

# Load the dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

# Visualize class distribution
label_counts = Counter(labels)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
plt.yscale('log')  # Apply logarithmic scale to y-axis
plt.xlabel('Labels (A-Z and Blank)', fontsize=12)
plt.ylabel('Count (Log Scale)', fontsize=12)
plt.title('Class Distribution in ASL Dataset', fontsize=16)
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()

# Inspect the data to ensure uniformity
max_length = max(len(d) for d in data)

# Pad sequences to the same length
padded_data = np.array([np.pad(d, (0, max_length - len(d)), 'constant') for d in data])

# Convert labels to integers
labels = np.asarray([int(label) for label in labels])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    padded_data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42, n_estimators=150)
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title("Confusion Matrix for ASL Recognition")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report
class_report = classification_report(y_test, y_predict, output_dict=True)
df_report = pd.DataFrame(class_report).transpose()

# Plot classification report as heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Classification Report for ASL Recognition")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.show()

# Visualize a single tree from the Random Forest
single_tree = model.estimators_[0]

# Create a visualization of the tree using dtreeviz
viz = dtreeviz(
    single_tree,
    x_train,  # Training data
    y_train,  # Training labels
    target_name="Class",  # Replace with your target's name
    feature_names=[f"Feature {i}" for i in range(x_train.shape[1])],
    class_names=[str(i) for i in np.unique(y_train)]
)

# Save the visualization to a file
viz.save("decision_tree.svg")

print("Decision tree saved as 'decision_tree.svg'")

# Save the trained model
# with open('asl_model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)

# print("ASL Recognition Model saved successfully!")
