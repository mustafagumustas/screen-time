import os
import cv2
import dlib
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from keras.preprocessing.image import ImageDataGenerator
from cv2 import FaceDetectorYN_create
from tqdm import tqdm

# Register HEIF opener for image conversion
register_heif_opener()


def convert_heic_to_jpg(filepath: str) -> np.ndarray:
    try:
        image = Image.open(filepath)
        jpg_path = os.path.splitext(filepath)[0] + ".jpg"
        image.save(jpg_path)
        return jpg_path
    except Exception as e:
        print(f"Error converting {filepath} to JPG: {e}")
        return None


def resize_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image: {image_path}")
            return None

        resized_img = cv2.resize(img, target_size)
        if resized_img is None:
            print(f"Error resizing image: {image_path}")
            return None

        return resized_img
    except Exception as e:
        print(f"Error processing image: {image_path}: {e}")
        return None


def align_faces_in_directory(directory: str, yunet_model_path: str) -> tuple:
    # Load the YUNET face detector model
    face_detector = FaceDetectorYN_create(yunet_model_path, "", (0, 0))

    aligned_faces = []
    image_names = []
    no_face_images = []
    no_face_image_names = []

    for filename in os.listdir(directory):
        if filename.startswith(".DS_Store"):
            continue
        filepath = os.path.join(directory, filename)

        # Convert HEIC to JPG if necessary
        if filepath.lower().endswith(".heic"):
            filepath = convert_heic_to_jpg(filepath)
            if filepath is None:
                continue

        # Resize the image to the required dimensions
        image = resize_image(filepath, target_size=(224, 224))
        if image is None:
            continue

        # Detect faces using YUNET
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        if len(faces) > 0:
            for face in faces:
                box = list(map(int, face[:4]))
                aligned_face = image[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
                aligned_faces.append(aligned_face)
                image_names.append(filename)
        else:
            print(f"No faces found in image: {filename}")
            no_face_images.append(image)
            no_face_image_names.append(filename)

    print(f"Aligned {len(aligned_faces)} faces and labeled {len(image_names)} faces.")
    print(f"Detected {len(no_face_images)} images with no faces.")
    return aligned_faces, image_names, no_face_images, no_face_image_names


def augment_data(folder_path):
    # Perform data augmentation and save the augmented images to the same folder
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image: {image_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((1,) + img.shape)  # Convert to 4D tensor
        i = 0
        for batch in data_gen.flow(
            img,
            batch_size=1,
            save_to_dir=folder_path,
            save_prefix="aug",
            save_format="jpg",
        ):
            i += 1
            if i >= 10:  # Generate 10 augmented images for each original image
                break


if __name__ == "__main__":
    data_folder = "data"
    yunet_model_path = "models/face_detection_yunet_2022mar.onnx"

    for person_folder in os.listdir(data_folder):
        person_folder_path = os.path.join(data_folder, person_folder)
        if os.path.isdir(person_folder_path):
            # Align faces in the person's folder using YUNET
            align_faces_in_directory(person_folder_path, yunet_model_path)

            # Perform data augmentation on the aligned faces
            augment_data(person_folder_path)
