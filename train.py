import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================
# 1. Dataset Paths
# =========================

train_dir = r"C:\Users\hp\Downloads\archive\train"
test_dir = r"C:\Users\hp\Downloads\archive\test"


# =========================
# 2. Image Settings
# =========================

IMG_SIZE = (48, 48)
BATCH_SIZE = 64


# =========================
# 3. Data Augmentation
# =========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# =========================
# 4. Training Data
# =========================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


# =========================
# 5. Validation Data
# =========================

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# =========================
# 6. Test Data
# =========================

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


print("\nClass Mapping:")
print(train_generator.class_indices)


# =========================
# 7. Lightweight CNN Model
# =========================

model = Sequential([

    # Block 1
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(48, 48, 1)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.20),

    # Block 2
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.30),

    # Block 4
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.30),

    # Reduce parameters
    GlobalAveragePooling2D(),

    # Fully Connected Layer
    Dense(128, activation="relu"),
    Dropout(0.40),

    # 7 Emotion Classes
    Dense(7, activation="softmax")
])


# =========================
# 8. Compile Model
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# 9. Show Model
# =========================

model.summary()


# =========================
# 10. Callbacks
# =========================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=0.00001
)


# =========================
# 11. Train Model
# =========================

print("\nStarting Training...\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# =========================
# 12. Test Model
# =========================

print("\nEvaluating Model on Test Dataset...\n")

test_loss, test_accuracy = model.evaluate(test_generator)

print("\n==============================")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("==============================")


# =========================
# 13. Save Model
# =========================

model.save("emotion_model.keras")

print("\nModel saved successfully!")
print("File: emotion_model.keras")