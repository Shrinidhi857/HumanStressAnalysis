import tensorflow as tf

# Load your Keras model
keras_model = tf.keras.models.load_model("emotion.keras")

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the converted model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
