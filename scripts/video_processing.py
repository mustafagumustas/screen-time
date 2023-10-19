import os
import dlib
import glob
import random
from multiprocessing import Process, Pool
import time
import sys
import concurrent.futures
import cv2
from pytube import YouTube

predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)


def extract_frames(output_dir, video_url):
    # Open the video file
    cap = cv2.VideoCapture(video_url)

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

    print(f"Frames with faces extracted: {frame_count % 25}")
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


def calculate_percentage_in_cluster(cluster_folder):
    # Initialize a dictionary to keep track of person counts
    person_counts = {}

    # Count the total number of faces in the cluster
    total_faces = 0

    for person_folder in os.listdir(cluster_folder):
        if os.path.isdir(os.path.join(cluster_folder, person_folder)):
            person_faces = len(os.listdir(os.path.join(cluster_folder, person_folder)))
            person_counts[person_folder] = person_faces
            total_faces += person_faces

    # Calculate the percentage of appearance for each person
    percentages = {}
    highest_appearance = 0  # Corrected variable name
    most_appeared_person = ""

    for person, count in person_counts.items():
        percentage = (count / total_faces) * 100
        percentages[person] = percentage

        if count > highest_appearance:
            highest_appearance = count
            most_appeared_person = person

    # Print the percentages
    for person, percentage in percentages.items():
        print(f"{person}: {percentage:.2f}%")

    print(f"Most appeared person: {most_appeared_person} ({highest_appearance} faces)")


def process_image_wrapper(args):
    # Unpack the arguments and call process_image
    image_subset, output_folder = args
    process_image(image_subset, output_folder)


if __name__ == "__main__":
    start_time = time.time()

    max_images_per_subset = 3000
    max_images_to_process = 100

    # Get the YouTube URL from the last command-line argument
    youtube_url = sys.argv[-1]

    # Initialize a YouTube object
    yt = YouTube(youtube_url)
    video_stream = yt.streams.get_highest_resolution()

    video_name = yt.title.replace(
        " ", "_"
    )  # Replace spaces with underscores in the video name
    faces_folder_path = f"{video_name}"
    output_folder_path = f"{video_name}"

    cap = cv2.VideoCapture(video_stream.url)

    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Extract frames from the video
    extract_frames(faces_folder_path, video_stream.url)

    # List all image paths in the faces folder
    all_image_paths = glob.glob(os.path.join(faces_folder_path, "*.jpg"))

    num_subsets = len(all_image_paths) // max_images_per_subset + 1
    random.shuffle(all_image_paths)
    image_subsets = [
        all_image_paths[i : i + max_images_per_subset]
        for i in range(0, len(all_image_paths), max_images_per_subset)
    ]

    # Create a Pool with a number of processes
    num_processes = 4  # Adjust the number of processes as needed
    with Pool(num_processes) as pool:
        # Prepare the arguments as tuples
        args = [
            (image_subset, os.path.join(output_folder_path, f"section_{i + 1}"))
            for i, image_subset in enumerate(image_subsets)
        ]
        pool.map(process_image_wrapper, args)

    # Print the percentages for each section
    for section_folder in os.listdir(output_folder_path):
        if os.path.isdir(os.path.join(output_folder_path, section_folder)):
            section_path = os.path.join(output_folder_path, section_folder)
            calculate_percentage_in_cluster(section_path)

    finish_time = time.time()
    print(f"Time taken: {finish_time - start_time:.2f} seconds")
