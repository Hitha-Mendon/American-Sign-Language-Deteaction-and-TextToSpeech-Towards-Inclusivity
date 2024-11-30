import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

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
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print('Confusion Matrix:')
print(conf_matrix)

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved successfully!")
