# import cv2
# import dlib
# import os
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing
# import sys
# import numpy as np
# face_counter = 0


# def print_progress(iteration, total, prefix="", suffix="", decimals=3, bar_length=100):
#     # Function to create a progress bar
#     format_str = "{0:." + str(decimals) + "f}"
#     percents = format_str.format(100 * (iteration / float(total)))
#     filled_length = int(round(bar_length * iteration / float(total)))
#     bar = "#" * filled_length + "-" * (bar_length - filled_length)
#     sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),
#     sys.stdout.flush()


# def detect_and_save_faces(frame, output_dir):
#     # Load the pre-trained face detector and shape predictor
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#     # Detect faces in the frame
#     faces = detector(frame)

#     for _, face in enumerate(faces):
#         # Get facial landmarks
#         landmarks = predictor(frame, face)

#         # Extract the coordinates of the facial landmarks
#         left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()

#         # Crop the face from the frame using the facial landmarks
#         face_image = frame[top:bottom, left:right]

#         # Save the cropped face as an image in the output directory with a unique name
#         # output_filename = os.path.join(output_dir, f"face_{face_counter}.jpg")
#         base, extension = os.path.splitext(output_dir)
#         cv2.imwrite(f"{base}_{_}{extension}", face_image)

#         # face_counter += 1  # Increment the counter for the next face


# # Load the pre-trained face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# source = "data/face_finder.mp4"
# output_dir = "output_faces"  # Directory to save the detected faces

# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)


# def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
#     """
#     Extract frames from a video using OpenCVs VideoCapture
#     :param video_path: path of the video
#     :param frames_dir: the directory to save the frames
#     :param overwrite: to overwrite frames that already exist?
#     :param start: start frame
#     :param end: end frame
#     :param every: frame spacing
#     :return: count of images saved
#     """

#     video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
#     frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

#     video_dir, video_filename = os.path.split(
#         video_path
#     )  # get the video path and filename from the path

#     assert os.path.exists(video_path)  # assert the video file exists

#     capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

#     if start < 0:  # if start isn't specified lets assume 0
#         start = 0
#     if end < 0:  # if end isn't specified assume the end of the video
#         end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

#     capture.set(1, start)  # set the starting frame of the capture
#     frame = start  # keep track of which frame we are up to, starting from start
#     while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
#     saved_count = 0  # a count of how many frames we have saved

#     while frame < end:  # lets loop through the frames until the end
#         _, image = capture.read()  # read an image from the capture

#         if while_safety > 500:  # break the while if our safety maxs out at 500
#             break

#         # sometimes OpenCV reads None's during a video, in which case we want to just skip
#         if (
#             image is None
#         ):  # if we get a bad return flag or the image we read is None, lets not save
#             while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
#             continue  # skip

#         if (
#             frame % every == 0
#         ):  # if this is a frame we want to write out based on the 'every' argument
#             while_safety = 0  # reset the safety count
#             save_path = os.path.join(
#                 frames_dir, video_filename, "{:010d}.jpg".format(frame)
#             )  # create the save path
#             if (
#                 not os.path.exists(save_path) or overwrite
#             ):  # if it doesn't exist or we want to overwrite anyways
#                 detect_and_save_faces(image, save_path)
#                 saved_count += 1  # increment our counter by one

#         frame += 1  # increment our frame count

#     capture.release()  # after the while has finished close the capture

#     return saved_count  # and return the count of the images we saved


# def video_to_frames(video_path, frames_dir, overwrite=False, every=1, chunk_size=1000):
#     """
#     Extracts the frames from a video using multiprocessing
#     :param video_path: path to the video
#     :param frames_dir: directory to save the frames
#     :param overwrite: overwrite frames if they exist?
#     :param every: extract every this many frames
#     :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
#     :return: path to the directory where the frames were saved, or None if fails
#     """

#     video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
#     frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

#     video_dir, video_filename = os.path.split(
#         video_path
#     )  # get the video path and filename from the path

#     # make directory to save frames, its a sub dir in the frames_dir with the video name
#     os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

#     capture = cv2.VideoCapture(video_path)  # load the video
#     total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
#     capture.release()  # release the capture straight away

#     if total < 1:  # if video has no frames, might be and opencv error
#         print("Video has no frames. Check your OpenCV + ffmpeg installation")
#         return None  # return None

#     frame_chunks = [
#         [i, i + chunk_size] for i in range(0, total, chunk_size)
#     ]  # split the frames into chunk lists
#     frame_chunks[-1][-1] = min(
#         frame_chunks[-1][-1], total - 1
#     )  # make sure last chunk has correct end frame, also handles case chunk_size < total

#     prefix_str = "Extracting frames from {}".format(
#         video_filename
#     )  # a prefix string to be printed in progress bar

#     # execute across multiple cpu cores to speed up processing, get the count automatically
#     with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
#         futures = [
#             executor.submit(
#                 extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every
#             )
#             for f in frame_chunks
#         ]  # submit the processes: extract_frames(...)

#         for i, f in enumerate(as_completed(futures)):  # as each process completes
#             print_progress(
#                 i, len(frame_chunks) - 1, prefix=prefix_str, suffix="Complete"
#             )  # print it's progress

#     return os.path.join(
#         frames_dir, video_filename
#     )  # when done return the directory containing the frames

# def generate_face_embeddings(face_image):
#     # Generate the face embedding for a given face image
#     face_encoding = face_rec_model.compute_face_descriptor(face_image)
#     return np.array(face_encoding)

# if __name__ == "__main__":
#     if sys.platform == "darwin":
#         multiprocessing.set_start_method("spawn")

#     # video_to_frames(
#     #     video_path="data/face_finder.mp4",
#     #     frames_dir="datatest_frames",
#     #     overwrite=False,
#     #     every=5,
#     #     chunk_size=1000,
#     # )


#     # Load the pre-trained face recognition model (dlib's model)
#     face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")


#     # Example usage:
#     face_image = cv2.imread("path_to_face_image.jpg")
#     face_embedding = generate_face_embeddings(face_image)


#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool for clustering using chinese_whispers.
#   This is useful when you have a collection of photographs which you know are linked to
#   a particular person, but the person may be photographed with multiple other people.
#   In this example, we assume the largest cluster will contain photos of the common person in the
#   collection of photographs. Then, we save extracted images of the face in the largest cluster in
#   a 150x150 px format which is suitable for jittering and loading to perform metric learning (as shown
#   in the dnn_metric_learning_on_images_ex.cpp example.
#   https://github.com/davisking/dlib/blob/master/examples/dnn_metric_learning_on_images_ex.cpp
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import os
import dlib
import glob
import random
import concurrent.futures

if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "   ./face_clustering.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces output_folder\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    )
    exit()


# Function to split a list into smaller sublists
def split_list(input_list, chunk_size):
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]
output_folder_path = sys.argv[4]
max_images_per_subset = 2000

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Limit the number of images processed
max_images_to_process = 100  # Set the maximum number of images to process

# Shuffle the list of image paths and select a subset
all_image_paths = glob.glob(os.path.join(faces_folder_path, "*.jpg"))

num_subsets = len(all_image_paths) // max_images_per_subset + 1
random.shuffle(all_image_paths)
image_subsets = [
    all_image_paths[i : i + max_images_per_subset]
    for i in range(0, len(all_image_paths), max_images_per_subset)
]


def process_image(image_path):
    img = dlib.load_rgb_image(image_path)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))


descriptors = []
images = []

# Process each subset separately
for i, image_subset in enumerate(image_subsets):
    print(f"Processing Subset {i + 1} of {num_subsets}")
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
        face_clusters[f"person_{j + 1}"] = []

    # Group faces by person
    for j, label in enumerate(labels):
        person_key = f"person_{label + 1}"
        img, shape = images[j]
        face_clusters[person_key].append((img, shape))

    # Ensure the output directory exists for this subset
    output_subset_folder = os.path.join(output_folder_path, f"section_{i + 1}")
    if not os.path.isdir(output_subset_folder):
        os.makedirs(output_subset_folder)

    # Save the face clusters to output folder
    for person, faces in face_clusters.items():
        if len(faces) >= 100:  # Ignore clusters with fewer than 100 faces
            person_folder = os.path.join(output_subset_folder, person)
            if not os.path.isdir(person_folder):
                os.makedirs(person_folder)

            for j, (img, shape) in enumerate(faces):
                file_path = os.path.join(person_folder, f"face_{j}.jpg")
                dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
