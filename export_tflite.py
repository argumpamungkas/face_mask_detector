import tensorflow as tf

MODEL_PATH = "face_mask_model.keras"

OUTPUT_PATH = "face_mask.tflite"

print("LOADING MODEL...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to tensorflow lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as file:
    file.write(tflite_model)

print()
print("======================")
print("TFLITE EXPORT FINISHED")
print("======================")

print(f"Output : {OUTPUT_PATH}")

print(f"Size : {len(tflite_model) / 1024:.2f} KB")
