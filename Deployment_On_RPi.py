import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("cnn_sandalwood.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open("cnn_sandalwood.tflite", "wb") as f:
    f.write(tflite_model)

import tensorflow.lite as tflite
import cv2
import numpy as np

# Load TFLite Model
interpreter = tflite.Interpreter(model_path="cnn_sandalwood.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Image
image = cv2.imread("test_leaf.jpg")
image = cv2.resize(image, (128, 128))
image = np.expand_dims(image, axis=0) / 255.0

# Run inference
interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

# Print Result
print(f"Disease Probability: {prediction[0][0]}")
