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
    image = Image.open(filepath)
    return np.array(image)


def load_images(directory: str) -> List[np.ndarray]:
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath.lower().endswith(".heic"):
            image = convert_heic_to_png(filepath)
        else:
            image = cv2.imread(filepath)
        if image is None:
            print(f"Unable to read image: {filename}")
            continue
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
            no_face_images.append(original_image)
            no_face_image_names.append(os.listdir(directory)[i])

    return aligned_faces, image_names, no_face_images, no_face_image_names


def augment_images(images: List[np.ndarray]) -> List[np.ndarray]:
    augmented_images = []
    for image in images:
        image = tf.convert_to_tensor(image)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        image = image.numpy()

        augmented_images.append(image)

    return augmented_images


def load_images_with_labels(base_directory: str) -> tuple:
    """Load images from the specified directory along with their labels."""
    images = []
    labels = []
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                filepath = os.path.join(subdir_path, filename)
                # Convert HEIC to numpy array if necessary, else load image into a numpy array
                if filepath.lower().endswith(".heic"):
                    image = convert_heic_to_png(filepath)
                else:
                    image = cv2.imread(filepath)
                if image is None:
                    print(f"Unable to read image: {filename}")
                    continue
                # Resize the image to the required dimensions
                image = resize_image(image)
                images.append(image)
                # Label is the name of the subdirectory
                labels.append(subdir)
    # Convert list of images to a numpy array
    images = np.array(images)
    return images, labels


def resize_image(image: np.ndarray, size=(224, 224)) -> np.ndarray:
    """Resize an image to the specified size."""
    return cv2.resize(image, size)
