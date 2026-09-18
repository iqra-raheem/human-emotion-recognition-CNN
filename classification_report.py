import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

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

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=emotions,
    output_dict=True
)

# Convert report to DataFrame
df = pd.DataFrame(report).transpose()

# Round values
df = df.round(4)

# Save CSV file
df.to_csv("classification_report.csv")

# Display report
print("\n========== CLASSIFICATION REPORT ==========\n")
print(df)

print("\nClassification report saved as classification_report.csv")