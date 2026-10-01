import tensorflow as tf

MODEL_PATH = "face_mask_model.keras"
OUTPUT_PATH = "face_mask_model.tflite"

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")

print("Input shape:")
print(model.input_shape)

print("Output shape:")
print(model.output_shape)


# ============================================================
# CONVERT
# ============================================================

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "wb"
) as f:

    f.write(tflite_model)


print()
print("========================================")
print("CONVERSION FINISHED")
print("========================================")

print(
    f"Model saved to: {OUTPUT_PATH}"
)