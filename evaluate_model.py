import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# 1. Model
# =========================

model = tf.keras.models.load_model("emotion_model.keras")


# =========================
# 2. Test Dataset
# =========================

test_dir = r"C:\Users\hp\Downloads\archive\test"

IMG_SIZE = (48, 48)
BATCH_SIZE = 64


test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# =========================
# 3. Emotion Names
# =========================

emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# =========================
# 4. Evaluate Model
# =========================

print("\nEvaluating model...\n")

loss, accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print("\n==============================")
print("Overall Test Accuracy:")
print(f"{accuracy * 100:.2f}%")
print("==============================")


# =========================
# 5. Predictions
# =========================

print("\nGenerating predictions...\n")

predictions = model.predict(
    test_generator,
    verbose=1
)


y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_generator.classes


# =========================
# 6. Classification Report
# =========================

print("\n==============================")
print("Emotion-wise Performance")
print("==============================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=emotions,
        digits=4
    )
)


# =========================
# 7. Confusion Matrix
# =========================

print("\n==============================")
print("Confusion Matrix")
print("==============================\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)