import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
file_path = r"D:\app\sign-language-detector-python-master\data.pickle"
data_dict = pickle.load(open(file_path, 'rb'))

# Initialize lists for valid data and labels
valid_data = []
valid_labels = []

# Inspect and preprocess data
target_length = 100  # Set a fixed length for feature vectors

for i, item in enumerate(data_dict['data']):
    try:
        array_item = np.array(item)
        if len(array_item.shape) == 1:  # Ensure it's a 1D array
            # Pad or truncate to a fixed length
            if len(array_item) < target_length:
                array_item = np.pad(array_item, (0, target_length - len(array_item)), mode='constant')
            elif len(array_item) > target_length:
                array_item = array_item[:target_length]
            valid_data.append(array_item)
            valid_labels.append(data_dict['labels'][i])
    except Exception as e:
        print(f"Skipping index {i}: {e}")

# Convert valid data and labels to NumPy arrays
data = np.array(valid_data)
labels = np.array(valid_labels)

# Check final data shape for debugging
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate the model's accuracy
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a file
output_path = r"D:\app\sign-language-detector-python-master\model.p"
with open(output_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Model saved to {output_path}.")
