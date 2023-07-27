import cv2
import dlib
import numpy as np
import os
from pillow_heif import register_heif_opener
from typing import List
from PIL import Image
import tensorflow as tf


def convert_heic_to_png(filepath: str) -> np.ndarray:
    register_heif_opener()
    print(f"Converting file: {filepath}")
    image = Image.open(filepath)
    return np.array(image)


def load_images(directory: str) -> List[np.ndarray]:
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath.startswith(".DS_Store"):
            continue
        if filepath.lower().endswith(".heic"):
            image = convert_heic_to_png(filepath)
        else:
            image = cv2.imread(filepath)
            print(f"Image read: {filepath}")
        if image is None:
            print(f"Unable to read image: {filename}")
            continue
        image = np.array(image, dtype=np.float32) / 255.0
        images.append(image)
    return images


def get_aligned_faces(directory: str, predictor_path: str) -> tuple:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    aligned_faces = []
    image_names = []
    no_face_images = []
    no_face_image_names = []
    images = load_images(directory)

    for i, image in enumerate(images):
        if image is None:
            print(f"Skipping face alignment for image {os.listdir(directory)[i]}")
            no_face_images.append(None)
            no_face_image_names.append(os.listdir(directory)[i])
            continue

        original_image = image.copy()
        no_faces = True
        for _ in range(4):
            dets = detector(image, 1)
            if len(dets) > 0:
                no_faces = False
                for k, d in enumerate(dets):
                    shape = predictor(image, d)
                    face_chip = dlib.get_face_chip(image, shape)
                    aligned_faces.append(face_chip)
                    image_names.append(os.listdir(directory)[i])
                break
            else:
                image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)

        if no_faces:
            print(f"No faces found in image {os.listdir(directory)[i]}")
            no_face_images.append(None)
            no_face_image_names.append(os.listdir(directory)[i])

    print(f"Aligned {len(aligned_faces)} faces and labeled {len(image_names)} faces.")
    print(f"Detected {len(no_face_images)} images with no faces.")
    return aligned_faces, image_names, no_face_images, no_face_image_names


def augment_images(images: List[np.ndarray]) -> List[np.ndarray]:
    augmented_images = []
    for image in images:
        if image is None:
            # Skip augmentation for None images
            continue
        image = tf.convert_to_tensor(image)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = image.numpy()  # Convert the Tensor back to a numpy array

        augmented_images.append(image)

    print(f"After augmentation, we have {len(augmented_images)} images.")
    return augmented_images


def load_images_with_labels(base_directory: str, target_size=(224, 224)) -> tuple:
    """Load images from the specified directory along with their labels."""
    images = []
    labels = []
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.startswith(".DS_Store"):
                    continue
                filepath = os.path.join(subdir_path, filename)
                # Convert HEIC to numpy array if necessary, else load image into a numpy array
                if filepath.lower().endswith(".heic"):
                    image = convert_heic_to_png(filepath)
                else:
                    image = cv2.imread(filepath)
                    if image is None:
                        print(f"Unable to read image: {filepath}")
                        continue
                    print(f"Image read: {filepath}")
                # Resize the image to the required dimensions
                image = resize_image(image, target_size)
                images.append(image)
                # Label is the name of the subdirectory
                labels.append(subdir)

    # Add "unknown" label to the labels list
    for i in range(len(images) - len(labels)):
        labels.append("unknown")

    # Check if the number of images and labels match
    if len(images) != len(labels):
        print("Error: Number of images and labels do not match!")
        print(f"Loaded {len(images)} images and {len(labels)} labels.")
        for subdir in os.listdir(base_directory):
            subdir_path = os.path.join(base_directory, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.startswith(".DS_Store"):
                        continue
                    filepath = os.path.join(subdir_path, filename)
                    if filepath not in labels:
                        print(f"Missing label for image: {filepath}")
        raise AssertionError("Number of images and labels do not match!")

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return images, labels


def resize_image(image: np.ndarray, size=(224, 224)) -> np.ndarray:
    """Resize an image to the specified size."""
    return cv2.resize(image, size)
