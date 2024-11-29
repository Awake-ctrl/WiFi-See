import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.ndimage import uniform_filter1d

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

# Function to preprocess CSI data (same as during training)
def preprocess_csi(csi_data, max_length=128):
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
    
    return csi_data

# Load the model
model = load_model('wifi_human_presence_model_with_preprocessing.h5')

# Load and preprocess new data for prediction
def load_and_preprocess_new_data(file_path, max_length=128):
    # Load the new CSV file
    df = pd.read_csv(file_path)
    
    # Preprocess the CSI data (same as the training data)
    csi_data = preprocess_csi(df["CSI_DATA"], max_length=max_length)
    
    # Reshape data for 2D-CNN (assuming single channel CSI data)
    csi_data = csi_data.reshape(-1, max_length, 1, 1)
    
    return csi_data

# Make predictions on new data
new_data_file_path = 'B247_saveri_karthik_false.csv'  # Replace with your new CSV file path
new_data = load_and_preprocess_new_data(new_data_file_path)

# Predict using the trained model
predictions = model.predict(new_data)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Display predictions for the first few samples
print("Predictions for the first few samples:", predicted_labels[:10])

# Assuming you have true labels (e.g., from the new dataset) to compare predictions with
# For demonstration purposes, let's create some mock true labels (replace with actual)
true_labels = np.random.randint(0, 2, size=len(predicted_labels))  # Random true labels for demonstration

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Human", "Human"], yticklabels=["No Human", "Human"])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
