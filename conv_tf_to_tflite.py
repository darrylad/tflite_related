import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model(r"C:\Users\kvv91\OneDrive\Desktop\BTP\VS code\leaf_disease_model_01.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open(r'C:\Users\kvv91\OneDrive\Desktop\BTP\VS code\leaf_disease_model_01.tflite', 'wb') as f:
    f.write(tflite_model)
