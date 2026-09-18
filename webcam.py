import cv2
import numpy as np
import csv
import time
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("emotion_model.keras")

# Emotion names
emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Start webcam
camera = cv2.VideoCapture(0)

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Store recent predictions for stability
prediction_history = []

# History file
history_file = "emotion_history.csv"
last_saved_time = 0

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera open nahi ho raha.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize for CNN
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        # Prepare input
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # CNN prediction
        prediction = model.predict(face, verbose=0)[0]

        # Add prediction to history
        prediction_history.append(prediction)

        # Keep last 5 predictions
        if len(prediction_history) > 5:
            prediction_history.pop(0)

        # Average predictions for stable result
        average_prediction = np.mean(
            prediction_history,
            axis=0
        )

        # Find emotion
        emotion_index = np.argmax(average_prediction)
        emotion = emotions[emotion_index]

        # Calculate confidence
        confidence = average_prediction[emotion_index] * 100

        # Save history every 2 seconds
        current_time = time.time()

        if current_time - last_saved_time >= 2:

            with open(history_file, "a", newline="") as file:

                writer = csv.writer(file)

                # Add headings to new file
                if file.tell() == 0:
                    writer.writerow([
                        "Date",
                        "Time",
                        "Emotion",
                        "Confidence"
                    ])

                writer.writerow([
                    time.strftime("%Y-%m-%d"),
                    time.strftime("%H:%M:%S"),
                    emotion,
                    f"{confidence:.2f}%"
                ])

            last_saved_time = current_time

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Show emotion
        cv2.putText(
            frame,
            emotion,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        # Show confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # Show webcam
    cv2.imshow(
        "Human Emotion Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
camera.release()
cv2.destroyAllWindows()

print("Webcam closed.")
print("Emotion history saved in emotion_history.csv")