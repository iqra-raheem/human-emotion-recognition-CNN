import pandas as pd
import matplotlib.pyplot as plt

# Read emotion history
data = pd.read_csv("emotion_history.csv")

# Count each emotion
emotion_counts = data["Emotion"].value_counts()

# Show counts in terminal
print("\nEmotion History:")
print(emotion_counts)

# Create graph
emotion_counts.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Emotion Recognition History")
plt.xlabel("Emotion")
plt.ylabel("Number of Predictions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()