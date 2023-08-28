import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.applications import MobileNetV2

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
directory = "data"

# Define data augmentation transformations for both training and validation
data_augmentation = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,  # Split data into training and validation
)

# Create training dataset using data augmentation
train_dataset_augmented = data_augmentation.flow_from_directory(
    directory,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=42,
)

# Create validation dataset using data augmentation
validation_dataset = data_augmentation.flow_from_directory(
    directory,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=42,
)

# Load MobileNetV2 base model
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)

# Fine-tune from this layer onwards
fine_tune_at = 120

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Add custom classification head
inputs = Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation="sigmoid")(x)

# Create the model
model = Model(inputs, outputs)

# Define BinaryCrossentropy loss function. Use from_logits=True
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define Adam optimizer with a learning rate of 0.1 * base_learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * 0.001)

# Use accuracy as evaluation metric
metrics = ["accuracy"]

# Compile the model
model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

# Train the model
history = model.fit(
    train_dataset_augmented,
    validation_data=validation_dataset,
    epochs=10,  # Adjust as needed
    verbose=1,
)

# Plot training history
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.legend(loc="lower right")
plt.show()

model.save("face_recognition_model_4.h5")

print("Model saved to disk.")
