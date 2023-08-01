import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
from data_preprocessing import resize_image


def main():
    data_folder = "data"

    # Step 1: Load images and labels from the data folder
    images, labels = load_images_with_labels(data_folder)

    # Step 2: Label Encoding with "unknown" for other people
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels_encoded = label_encoder.transform(labels)

    # Step 3: Data Split (70% train, 15% validation, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Step 4: Preprocess the images (resize to the appropriate input size for MobileNetV2)
    target_size = (224, 224)
    X_train = np.array([resize_image(img, target_size) for img in X_train])
    X_val = np.array([resize_image(img, target_size) for img in X_val])
    X_test = np.array([resize_image(img, target_size) for img in X_test])

    # Step 5: Model Architecture (MobileNetV2 with custom head)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(len(label_encoder.classes_), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Step 6: Compile the model
    model.compile(
        optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # One-hot encode the labels for categorical_crossentropy
    y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_val = to_categorical(y_val, num_classes=len(label_encoder.classes_))
    y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

    # Define callbacks (e.g., ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
    checkpoint_path = "model_checkpoint.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        verbose=1,
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
    )

    # Step 7: Model Training
    batch_size = 32
    epochs = 20

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop, reduce_lr],
    )

    # Step 8: Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Step 9: Save the final model
    model.save("face_recognition_model.h5")


if __name__ == "__main__":
    main()
