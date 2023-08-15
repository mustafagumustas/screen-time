import os
import cv2
import dlib
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
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


def align_faces_in_directory(
    directory: str, yunet_model_path: str, aligned_faces_dir: str
) -> tuple:
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
            for idx, face in enumerate(faces):
                box = list(map(int, face[:4]))
                aligned_face = image[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]

                # Save the aligned face to the aligned_faces_dir with a unique name
                aligned_face_path = os.path.join(
                    aligned_faces_dir, f"{filename[:-4]}_face_{idx}.jpg"
                )
                cv2.imwrite(aligned_face_path, aligned_face)

                aligned_faces.append(aligned_face)
                image_names.append(filename)

            # Move the original image to another folder (optional)
            original_image_path = os.path.join(directory, filename)
            os.rename(
                original_image_path,
                os.path.join(directory, "original_images", filename),
            )
        else:
            print(f"No faces found in image: {filename}")
            no_face_images.append(image)
            no_face_image_names.append(filename)

    print(f"Aligned {len(aligned_faces)} faces and labeled {len(image_names)} faces.")
    print(f"Detected {len(no_face_images)} images with no faces.")
    return aligned_faces, image_names, no_face_images, no_face_image_names


def crop_and_save_faces(aligned_faces, image_names, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for face, image_name in zip(aligned_faces, image_names):
        face_save_path = os.path.join(
            save_dir, f"{os.path.splitext(image_name)[0]}_face.jpg"
        )
        cv2.imwrite(face_save_path, face)

    print(f"{len(aligned_faces)} faces cropped and saved in {save_dir} directory.")


if __name__ == "__main__":
    data_folder = "data"
    yunet_model_path = "models/face_detection_yunet_2022mar.onnx"

    for person_folder in os.listdir(data_folder):
        person_folder_path = os.path.join(data_folder, person_folder)
        if os.path.isdir(person_folder_path):
            # Align faces in the person's folder using YUNET
            aligned_faces, image_names, _, _ = align_faces_in_directory(
                person_folder_path, yunet_model_path
            )

            # Save the aligned faces in a separate directory
            save_dir = os.path.join(data_folder, f"{person_folder}_cropped_faces")
            crop_and_save_faces(aligned_faces, image_names, save_dir)
