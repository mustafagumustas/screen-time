import cv2
import dlib
import numpy as np
from scripts.face_tools import face_degree
from PIL import Image
import os
from pillow_heif import register_heif_opener
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()

MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"
MAX_FRAME_COUNT = 15


def initialize_detector_predictor():
    predictor = dlib.shape_predictor(MODEL_PATH)
    return detector, predictor


def detect_faces(frame, detector):
    faces = detector(frame)
    return faces


def align_faces(frame, faces, predictor):
    aligned_faces = []
    for face in faces:
        landmarks = predictor(frame, face)
        landmarks = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        )
        aligned_face = face_degree(
            frame, landmarks
        )  # Assuming you have the face_degree function implemented
        aligned_faces.append(aligned_face)
    return aligned_faces


def crop_faces(frame, detector):
    cropped_faces = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for rect in faces:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cropped_face = frame[y1:y2, x1:x2]
        cropped_faces.append(cropped_face)
    return cropped_faces


def process_frames(max_frame_count):
    detector, predictor = initialize_detector_predictor()
    cap = cv2.VideoCapture(0)
    frame_count = 0  # Counter for the saved frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, detector)
        aligned_faces = align_faces(frame, faces, predictor)
        cropped_faces = crop_faces(aligned_faces)

        for cropped_face in cropped_faces:
            resized_face = cv2.resize(
                cropped_face, (224, 224)
            )  # Resize the face to 224x224
            cv2.imshow("Do you want to save this?", resized_face)

            key = cv2.waitKey(0)
            if key == ord("s"):  # Press 's' to save the current frame
                frame_count += 1
                cv2.imwrite(f"data/new_face/face_{frame_count}.jpg", resized_face)
                if frame_count >= max_frame_count:
                    break
            else:
                continue

        if frame_count >= max_frame_count:
            break

        # display frame with detected faces
        frame = cv2.resize(frame, (224, 224))  # Resize the original frame to 224x224
        display_frame = np.hstack(
            (resized_face, frame)
        )  # Horizontally stack the resized face and the original frame
        cv2.imshow("Face Detection", display_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def read_images(path):
    imgs = []
    valid_images = [".jpg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            if "heic" in ext.lower():
                # if there is a heic image, convert it to jpg
                # and move it to the parent path
                register_heif_opener()
                image = Image.open(os.path.join(path, f)).rotate(-90)
                image.save(
                    os.path.join(path, f"{os.path.splitext(f)[0]}.jpg"), format="JPEG"
                )
                os.rename(
                    os.path.join(path, f), os.path.join(os.path.split(path)[0], f)
                )
                img_array = np.asarray(Image.open(os.path.join(path, f)))
                imgs.append(img_array)
            continue
        img_array = np.asarray(Image.open(os.path.join(path, f)))
        imgs.append(img_array)
    return imgs


def extract_and_show_faces(image_directory):
    detector, predictor = initialize_detector_predictor()
    images = read_images(image_directory)
    face_counter = 0  # Counter for the faces

    for image in images:
        faces = detect_faces(image, detector)
        aligned_faces = align_faces(image, faces, predictor)

        for face in aligned_faces:
            cropped_face = crop_faces(face, detector)[
                0
            ]  # Assuming only one face per aligned_face

            cv2.imwrite(
                f"face_{face_counter}.jpg",
                cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR),
            )
            face_counter += 1


# Example usage:
image_directory = "/Users/mustafagumustas/screen-time/data/new_face/val"
extract_and_show_faces(image_directory)
