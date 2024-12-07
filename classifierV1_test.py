import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('leaf_disease_model_01.h5')

# Load the class mapping from the JSON file
with open('class_mapping.json', 'r') as json_file:
    class_mapping = json.load(json_file)

# Define the image dimensions
img_width, img_height = 150, 150  # Change to the dimensions used during training

# Define the path to the folder containing test images
test_folder_path = r"C:\Users\kvv91\OneDrive\Desktop\J3\data10623\refined_dataset_26102023\test"  # Update with your actual path

# Get a list of all files in the folder
test_files = [os.path.join(test_folder_path, file) for file in os.listdir(test_folder_path)]

# Iterate over each test image
for test_file in test_files:
    # Load and preprocess the image
    img = image.load_img(test_file, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the normalization during training

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Print the predicted class and file name
    print(f'Test Image: {test_file} - Predicted Class Index: {predicted_class_index}')

    # Map the predicted class index to the specific class name
    predicted_class_label = class_mapping[str(predicted_class_index)]  # Use str() to ensure matching keys
    print(f'Test Image: {test_file} - Predicted Class Label: {predicted_class_label}')
