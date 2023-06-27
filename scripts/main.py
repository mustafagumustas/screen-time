from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from data_preprocessing import (
    convert_heic_to_png,
    load_images,
    get_aligned_faces,
    augment_images,
    load_images_with_labels,
)
import numpy as np

# Load and prepare data
images, labels = load_images_with_labels("data/new_face")
print(f"Loaded {len(images)} images and {len(labels)} labels.")

# Ensure images and labels have the same length
assert len(images) == len(labels)

# Convert labels to one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

images = augment_images(images)
images = np.array(images)
print(f"After augmentation, we have {len(images)} images and {len(labels)} labels.")

# Ensure images and labels have the same length
assert len(images) == len(labels)

# Determine the number of unique classes
num_classes = len(np.unique(labels))
print(f"Unique labels: {np.unique(labels)}")

# Encoding labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = labels.reshape(len(labels), 1)  # Reshape labels for one-hot encoding

# One-hot encoding
ohe = OneHotEncoder(sparse=False)
labels = ohe.fit_transform(labels)
print(f"Data cardinality before split: {len(images)} images, {len(labels)} labels.")

# Split into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, stratify=labels
)

# Load MobileNetV2 model without the top classification layer
base_model = MobileNetV2(weights="imagenet", include_top=False)
x = base_model.output  # Add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)  # Add a fully-connected layer
x = Dropout(0.5)(x)  # Add a dropout layer with a rate of 0.5
predictions = Dense(num_classes, activation="softmax")(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First, we will only train the top layers (which were randomly initialized)
# i.e. freeze all convolutional MobileNetV2 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    train_images, train_labels, epochs=10, validation_data=(val_images, val_labels)
)
model.save("face_recognition_model_260623_3.h5")
