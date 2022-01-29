import tensorflow as tf

model=tf.keras.models.load_model("/content/MaskNet")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model) 