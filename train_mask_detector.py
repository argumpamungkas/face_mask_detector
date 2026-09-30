import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# CONFIG

DATASET_DIR = "dataset"

IMG_SIZE = 160
BATCH_SIZE = 32 # digunakan untuk batch dari EPOCHS (32 gambar satu kali proses EPOCHS)
EPOCHS = 15

MODEL_OUTPUT = "face_mask_model.keras"

# LOAD DATASET
print("LOADING DATASET....")

train_dataset = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

validation_dataset = image_dataset_from_directory(
    DATASET_DIR,
    # data set dibagi 2 (80% Training, 20% Validation)
    # misal ada gambar 1000, maka 800 digunakan training dan 200 digunakan validasi
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

# CHECK CLASS

class_names = train_dataset.class_names

print()
print("CLASSES:")
print(class_names)

print()
print("CLASS MAPPING:")

for index, name in enumerate(class_names):
    print(index, "=", name)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


# ============================================================
# MODEL
# ============================================================

# Ini adalah Convolutional Neural Network (CNN)
# Biasanya digunakan untuk image classification.
# layers.Conv2D(
#         32,
#         (3, 3),
#         activation="relu"
#     ),

model = models.Sequential([
    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    data_augmentation,

    layers.Rescaling(
        1.0 / 255
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        2,
        activation="softmax"
    ),
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# SUMMARY
# ============================================================

model.summary()


# ============================================================
# TRAIN
# ============================================================

print()
print("Starting training...")
print()

# EPOCHS adalah dataset training yang akan dipelajari sebanyak 15 putaran.
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
)


# ============================================================
# SAVE
# ============================================================

model.save(
    MODEL_OUTPUT
)

print()
print("======================================")
print("TRAINING FINISHED")
print("======================================")

print(
    f"Model saved to: {MODEL_OUTPUT}"
)


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    "accuracy.png"
)

print(
    "Accuracy graph saved to accuracy.png"
)