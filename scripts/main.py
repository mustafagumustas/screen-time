from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.mobilenet_v2 import preprocess_input


# Define paths to your training and validation image folders
train_data_dir = "data/training_data"
validation_data_dir = "data/validation_data"

# Set image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Validation data generator (no augmentation)
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255, preprocessing_function=preprocess_input
)

# Create data iterators from generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",  # Change to 'binary' if you have 2 classes
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",  # Change to 'binary' if you have 2 classes
)

# Load MobileNetV2 without top layers and with imagenet weights
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)


# Add custom layers on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)  # Add dropout layer here
predictions = Dense(3, activation="softmax")(
    x
)  # Replace num_classes with the number of your classes

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)


# Freeze the layers of the pre-trained base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
)

# Print a summary of the model architecture
# model.summary()


callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True),
    EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor="val_loss",
    ),
    ReduceLROnPlateau(factor=0.2, patience=5),
]

epochs = 50


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks,
)


model.save("face_recognition_model_300823.h5")
