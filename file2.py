import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical

# Define the path to the folder containing CSV files
DATA_FOLDER = "../koutput"

# Hampel filter for noise removal
def hampel_filter(data, window_size=5, n_sigmas=3):
    n = len(data)
    filtered_data = data.copy()
    for i in range(window_size, n - window_size):
        window = data[i - window_size:i + window_size + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        threshold = n_sigmas * mad
        if np.abs(data[i] - median) > threshold:
            filtered_data[i] = median
    return filtered_data

# Function to preprocess CSI data
def preprocess_csi(csi_data, max_length=128, apply_pca=False):
    # Convert CSI string to array and apply padding/truncation
    csi_data = csi_data.apply(lambda x: np.array([int(i) for i in x.strip("[]").split()]))
    csi_data = csi_data.apply(lambda x: np.pad(x, (0, max(0, max_length - len(x))), mode='constant')[:max_length])
    
    # Apply Hampel filter for outlier removal
    csi_data = csi_data.apply(hampel_filter)
    
    # Apply moving average filter
    csi_data = csi_data.apply(lambda x: uniform_filter1d(x, size=5))
    
    # Stack into a NumPy array
    csi_data = np.stack(csi_data.values)
    
    # Normalize the data
    csi_data = csi_data / np.max(csi_data)
    
    # Apply PCA (optional)
    if apply_pca:
        pca = PCA(n_components=32)  # Reduce to 32 components
        csi_data = pca.fit_transform(csi_data)
    
    return csi_data

def load_data(data_folder, max_length=128):
    data_frames = []
    
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder, file)
            
            # Open the file to check problematic line before loading it
            with open(file_path, 'r', encoding='utf-8', errors='replace') as fileee:
                lines = fileee.readlines()
                if len(lines) > 33856:
                    print(f"Problematic line at index 33856 in file {file_path}: {lines[33856]}")

            try:
                # Try loading with UTF-8 encoding
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # If a UnicodeDecodeError occurs, fallback to ISO-8859-1 encoding
                print(f"Error decoding {file_path} with UTF-8. Falling back to ISO-8859-1 encoding.")
                df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')
            
            data_frames.append(df)
    
    # Concatenate all the data frames into a single data frame
    data = pd.concat(data_frames, ignore_index=True)
    
    # Extract CSI data and labels (assuming "signal" is the label for human presence detection)
    csi_data = preprocess_csi(data["CSI_DATA"], max_length=max_length, apply_pca=False)
    labels = data["signal"].astype(int)
    
    # Reshape data for 1D-CNN (assuming single channel CSI data)
    csi_data = csi_data.reshape(-1, max_length, 1)  # (samples, time_steps, features)
    
    return csi_data, labels

# Load the data
csi_data, labels = load_data(DATA_FOLDER)

# One-hot encode labels
labels = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(csi_data, labels, test_size=0.2, random_state=42)

# Build the Keras Sequential model using Conv1D
def build_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),  # Conv1D for sequence data
        MaxPooling1D(2),  # Pooling along the time dimension
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(labels.shape[1], activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the model
input_shape = (csi_data.shape[1], csi_data.shape[2])  # (time_steps, features)
model = build_model(input_shape)
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model.save("wifi_human_presence_model_with_preprocessing.h5")

# Prediction example
sample_index = 0  # Replace with a valid index
sample_data = X_test[sample_index].reshape(1, *input_shape)
prediction = model.predict(sample_data)
print(f"Predicted label: {np.argmax(prediction)} | True label: {np.argmax(y_test[sample_index])}")
