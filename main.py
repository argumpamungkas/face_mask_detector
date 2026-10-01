import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "face_mask_model.keras"

FACE_CASCADE = "haarcascade/haarcascade_frontalface_default.xml"

IMG_SIZE = 160

FACE_CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# CLASS
# ============================================================

CLASS_NAMES = [
    "mask",
    "no_mask",
]


# ============================================================
# LOAD MASK MODEL
# ============================================================

print("Loading mask model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Mask model loaded.")


print()
print("Classes:")

for index, name in enumerate(CLASS_NAMES):

    print(
        index,
        "=",
        name
    )


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

print()
print("Loading face detector...")

face_detector = cv2.CascadeClassifier(
    FACE_CASCADE
)


# Pastikan XML berhasil dibaca

if face_detector.empty():

    print()
    print("ERROR:")
    print(
        "Face detector tidak berhasil dibuka."
    )

    print()
    print(
        f"Pastikan file '{FACE_CASCADE}'"
    )

    print(
        "berada satu folder dengan main.py."
    )

    exit()


print("Face detector loaded.")


# ============================================================
# CAMERA
# ============================================================

print()
print("Opening camera...")

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Camera tidak dapat dibuka."
    )

    exit()


print()
print("========================================")
print("Camera started")
print("Press Q to exit")
print("========================================")


# ============================================================
# LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print(
            "ERROR: Tidak dapat membaca frame."
        )

        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # ========================================================
    # NO FACE
    # ========================================================

    if len(faces) == 0:

        cv2.putText(
            frame,
            "NO FACE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # PROCESS FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # ----------------------------------------------------
        # CROP FACE
        # ----------------------------------------------------

        face = frame[
            y:y + h,
            x:x + w
        ]


        if face.size == 0:

            continue


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        face_resized = cv2.resize(
            face,
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        face_rgb = cv2.cvtColor(
            face_resized,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # NUMPY
        # ----------------------------------------------------

        image_array = np.array(
            face_rgb,
            dtype=np.float32
        )


        # Add batch dimension

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            image_array,
            verbose=0
        )


        # ----------------------------------------------------
        # CLASS
        # ----------------------------------------------------

        class_index = np.argmax(
            prediction[0]
        )


        confidence = float(
            prediction[0][class_index]
        )


        class_name = CLASS_NAMES[
            class_index
        ]


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"{class_name.upper()} "
            f"{confidence * 100:.1f}%"
        )


        # ----------------------------------------------------
        # DRAW FACE
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # DRAW LABEL
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (
                x,
                max(
                    30,
                    y - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        mask_probability = float(
            prediction[0][0]
        )


        no_mask_probability = float(
            prediction[0][1]
        )


        debug_text = (
            f"mask={mask_probability:.2f} "
            f"no_mask={no_mask_probability:.2f}"
        )


        cv2.putText(
            frame,
            debug_text,
            (
                x,
                y + h + 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Face Mask Detection",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Camera stopped.")