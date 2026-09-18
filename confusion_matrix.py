import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Model load
model = tf.keras.models.load_model("emotion_model.keras")

# Test dataset
test_dir = r"C:\Users\hp\Downloads\archive\test"

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    shuffle=False
)

# Emotion names
emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Normalize images
test_data = test_data.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
)

# Actual labels
y_true = np.concatenate([y.numpy() for x, y in test_data])

# Predictions
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Draw confusion matrix
plt.figure(figsize=(9, 7))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Human Emotion Recognition")
plt.colorbar()

plt.xticks(
    np.arange(len(emotions)),
    emotions,
    rotation=45
)

plt.yticks(
    np.arange(len(emotions)),
    emotions
)

# Numbers inside cells
for i in range(len(emotions)):
    for j in range(len(emotions)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()

# Save image
plt.savefig("confusion_matrix.png", dpi=300)

print("\nConfusion matrix saved as confusion_matrix.png")

plt.show()