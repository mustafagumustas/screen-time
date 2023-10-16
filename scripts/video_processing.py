import os
import dlib
import glob
import random
from multiprocessing import Process
import time
import sys
import concurrent.futures
import cv2

predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)
# example frmae
# /Users/mustafagumustas/screen-time/output_faces/frame_0375.jpg


def extract_frames(video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Save frames with detected faces at the specified interval
        if frame_count % fps == 0:
            faces = detector(frame)
            if len(faces) >= 1:
                output_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(output_file, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames with faces extracted: {frame_count% 25}")
    print(f"Frames with faces saved in: {output_dir}")


def generate_face_embeddings(face_image):
    image = cv2.imread(face_image)
    if image is None:
        print(f"Error: Could not read image from {face_image}")
        return None  # Handle the error gracefully

    if image.size == 0:
        print(f"Error: Empty image from {face_image}")
        return None  # Handle the error gracefully

    image = cv2.resize(image, (150, 150))
    # Compute the face descriptor for the face chip
    face_encoding = facerec.compute_face_descriptor(image)

    return face_encoding


def cluster_faces(face_embeddings):
    labels = dlib.chinese_whispers_clustering(face_embeddings, threshold=0.6)
    return labels


def process_image(image_subset, output_folder):
    descriptors = []
    images = []

    for f in image_subset:
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            descriptors.append(face_descriptor)
            images.append((img, shape))

    labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
    num_classes = len(set(labels))
    print("Number of clusters: {}".format(num_classes))

    # Create a dictionary to store faces for each person
    face_clusters = {}
    for j in range(num_classes):
        person_key = f"person_{j + 1}"
        face_clusters[person_key] = []

    # Group faces by person
    for j, label in enumerate(labels):
        person_key = f"person_{label + 1}"
        img, shape = images[j]
        face_clusters[person_key].append((img, shape))

    # Ensure the output directory exists for this subset
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Save the face clusters to output folder
    for person, faces in face_clusters.items():
        if len(faces) >= 10:  # Ignore clusters with fewer than 10 faces
            person_folder = os.path.join(output_folder, person)
            if not os.path.isdir(person_folder):
                os.makedirs(person_folder)

            for j, (img, shape) in enumerate(faces):
                file_path = os.path.join(person_folder, f"face_{j}.jpg")
                dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)


if __name__ == "__main__":
    start_time = time.time()
    video_path = "avrupayakasi.mp4"
    faces_folder_path = "output_faces"
    output_folder_path = "output_clusters"
    max_images_per_subset = 2000
    max_images_to_process = 100

    # Extract frames from the video
    extract_frames(video_path, faces_folder_path)

    # List all image paths in the faces folder
    all_image_paths = glob.glob(os.path.join(faces_folder_path, "*.jpg"))

    num_subsets = len(all_image_paths) // max_images_per_subset + 1
    random.shuffle(all_image_paths)
    image_subsets = [
        all_image_paths[i : i + max_images_per_subset]
        for i in range(0, len(all_image_paths), max_images_per_subset)
    ]

    processes = []

    for i, image_subset in enumerate(image_subsets):
        output_subset_folder = os.path.join(output_folder_path, f"section_{i + 1}")
        p = Process(target=process_image, args=(image_subset, output_subset_folder))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    finish_time = time.time()
    print(f"Time taken: {finish_time - start_time:.2f} seconds")
