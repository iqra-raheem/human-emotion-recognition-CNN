# Human Emotion Recognition Using CNN
##  Project Overview

Human Emotion Recognition Using CNN is an AI-based system that
detects facial expressions and classifies them into seven emotion
categories.

The system uses a Convolutional Neural Network (CNN) trained on
the FER-2013 dataset. It supports real-time emotion recognition
through a webcam and emotion prediction from uploaded images.

A Flask-based web dashboard is used to display predictions,
confidence scores, emotion history, graphs, and model evaluation
results.
##  Objectives

- Develop a CNN-based human emotion recognition system.
- Recognize seven different human emotions.
- Provide real-time emotion detection through a webcam.
- Allow users to upload images for emotion prediction.
- Display prediction confidence and analysis results.

##  Emotion Classes

Our system recognizes 7 emotions:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

##  Dataset

We used the FER-2013 (Facial Expression Recognition 2013) dataset.

- Total Images: 35,887
- Training Images: 28,709
- Test Images: 7,178
- Image Size: 48 × 48 pixels
- Image Type: Grayscale
- Emotion Classes: 7

##  CNN Model

Our system uses a Convolutional Neural Network (CNN) to recognize
human emotions from facial images.

### Model Architecture

Input: 48 × 48 × 1 Grayscale Image

↓
Conv2D (32 filters) + Batch Normalization + Max Pooling + Dropout

↓
Conv2D (64 filters) + Batch Normalization + Max Pooling + Dropout

↓
Conv2D (128 filters) + Batch Normalization + Max Pooling + Dropout

↓
Conv2D (128 filters) + Batch Normalization + Max Pooling + Dropout

↓
Global Average Pooling

↓
Dense (128 neurons) + Dropout

↓
Softmax Output (7 Emotions)

### Model Details

- Total Parameters: 259,079
- Optimizer: Adam
- Learning Rate: 0.001
- Input Size: 48 × 48 × 1
- Output Classes: 7
- Model File: emotion_model.keras

##  Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Flask
- HTML
- CSS

##  Features

- 🔐 User Login and Logout
- 📷 Real-time Emotion Detection using Webcam
- 👥 Multiple Face Detection
- 🖼️ Image Upload for Emotion Prediction
- 📊 Emotion Confidence Percentage
- 📈 Emotion Analysis Graph
- 📝 Emotion History
- 📋 Classification Report
- 🔲 Confusion Matrix
- 🌐 Flask Web Dashboard

## System Workflow

The system works through the following steps:

1. Webcam or Image Upload
2. Face Detection
3. Grayscale Conversion
4. Resize Image to 48 × 48
5. CNN Model Processing
6. Emotion Probability Calculation
7. Select Highest Probability Emotion
8. Display Emotion and Confidence
9. Store Prediction in Emotion History
10. Display Graph and Analysis

##  Model Results

The trained CNN model was evaluated on 7,178 test images.

### Test Accuracy

54.81%

### Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The results show that the model can recognize facial expressions,
but some emotions are difficult to classify because their facial
features can be similar.

##  Future Improvements

In the future, the system can be improved by:

- Using advanced CNN architectures.
- Applying transfer learning.
- Using a larger and more balanced dataset.
- Improving data augmentation techniques.
- Using better face detection and tracking.
- Improving real-time prediction speed.
- Developing a mobile application.
- Adding more emotion categories.

##  License

This project was developed for academic and educational purposes.

##  Author

**iqra raheem**

Human Emotion Recognition Using CNN
