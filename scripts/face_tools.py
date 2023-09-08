import cv2
import numpy as np
import math
from PIL import Image
import uuid
import os
import dlib
from keras_preprocessing.image import img_to_array
from keras.applications.resnet import preprocess_input
from cv2 import FaceDetectorYN_create
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

# landmarks locations in their list
# The mouth can be accessed through points [48, 68].
# The right eyebrow through points [17, 22].
# The left eyebrow through points [22, 27].
# The right eye using [36, 42].
# The left eye with [42, 48].
# The nose using [27, 35].
# And the jaw via [0, 17].
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


def euclidean_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


# could be dlib.get_face_chips used instead
def face_degree(frame, landmarks):
    right_eye = landmarks[42:48]
    left_eye = landmarks[36:42]
    nose = landmarks[27:35]

    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)
    nose_center = np.mean(nose, axis=0).astype(int)

    # find rotation direction
    if left_eye_center[1] > right_eye_center[1]:
        point_3rd = right_eye_center[0], left_eye_center[1]
        direction = -1  # rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = left_eye_center[0], right_eye_center[1]
        direction = 1  # rotate inverse direction of clock
        print("rotate to inverse clock direction")

    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, point_3rd)
    c = euclidean_distance(right_eye_center, left_eye_center)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    # print("cos(a) = ", cos_a)
    angle = np.arccos(cos_a)
    # print("angle: ", angle," in radian")

    angle = (angle * 180) / math.pi
    print("angle: ", angle, " in degree")

    if direction == -1:
        angle = 90 - angle

    print("angle: ", angle, " in degree")

    # rotate image
    new_img = Image.fromarray(frame)
    new_img = np.array(new_img.rotate(direction * angle))
    # cv2.imshow("Face Detection", new_img)
    return new_img


def save_frame(face: np.ndarray, save_dir: str) -> None:
    """
    Saves the face image to a specified directory.

    Args:
        face (np.ndarray): Face image.
        save_dir (str): Directory to save the image.
    """
    filename = str(uuid.uuid4()) + ".jpg"
    save_path = os.path.join(save_dir, filename)
    print(f"Image saved to: {save_path}")
    cv2.imwrite(save_path, face)


def resize_img(directory):
    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        # Get the full file path
        image_path = os.path.join(directory, filename)

        # Check if the file is an image (you can modify this condition based on your file types)
        if os.path.isfile(image_path) and filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            # Read the cropped face image
            cropped_face = cv2.imread(image_path)

            # Perform the preprocessing steps as mentioned earlier
            resized_face = cv2.resize(cropped_face, (224, 224))
            resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            preprocessed_face = preprocess_input(resized_face)
            return preprocessed_face


def reorder_images_under_folder(folder_path: str) -> None:
    for subfile in os.listdir(folder_path):
        subfile_path = os.path.join(folder_path, subfile)
        if os.path.isdir(subfile_path):
            reorder_images_under_folder(subfile_path)
        else:
            os.rename(
                subfile_path,
                os.path.join(folder_path, folder_path.split("/")[-1] + "_" + subfile),
            )


def process_images_in_folder(shape_predictor_path, image_folder_path):
    # Initialize the face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Initialize counters
    total_images = 0
    no_faces_images = 0
    multiple_faces_images = 0

    # Iterate through all images in the folder
    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            total_images += 1
            image_path = os.path.join(image_folder_path, filename)

            # Load the image
            image = dlib.load_rgb_image(image_path)

            # Detect faces in the image
            faces = detector(image)

            if len(faces) == 0:
                print(f"No faces detected in: {image_path}")
                no_faces_images += 1
            elif len(faces) > 1:
                print(f"Multiple faces detected in: {image_path}")
                multiple_faces_images += 1

    # Print the report
    print("------- Report -------")
    print(f"Total images processed: {total_images}")
    print(f"Images with no faces detected: {no_faces_images}")
    print(f"Images with multiple faces detected: {multiple_faces_images}")


def convert_heic_to_jpg(filepath: str) -> np.ndarray:
    try:
        image = Image.open(filepath)
        jpg_path = os.path.splitext(filepath)[0] + ".jpg"
        image.save(jpg_path)
        return jpg_path
    except Exception as e:
        print(f"Error converting {filepath} to JPG: {e}")


def align_faces_in_image(image_path, yunet_model_path):
    # Load the YUNET face detector model
    face_detector = FaceDetectorYN_create(yunet_model_path, "", (0, 0))

    aligned_faces = []
    image_names = []

    # Load the image
    image = cv2.imread(image_path)

    if image is not None:
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
                image_names.append("aligned_face")

    return aligned_faces, image_names


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, frame):
        # Convert the frame to grayscale (dlib requires grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = self.detector(gray)

        # Process the detected faces
        detected_faces = []
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            detected_faces.append((x1, y1, x2, y2))

        return detected_faces


def print_progress(iteration, total, prefix="", suffix="", decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(
        100 * (iteration / float(total))
    )  # calculate the % done
    filled_length = int(
        round(bar_length * iteration / float(total))
    )  # calculate the filled bar length
    bar = "#" * filled_length + "-" * (
        bar_length - filled_length
    )  # generate the bar string
    sys.stdout.write(
        "\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)
    ),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(
        video_path
    )  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    while frame < end:  # loop through the frames
        _, image = capture.read()  # read a frame from the video

        if while_safety > 500:
            break

        if image is None:
            while_safety += 1
            continue

        if frame % every == 0:
            while_safety = 0
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            faces = detector(gray)

            if len(faces) > 0:
                save_path = os.path.join(
                    frames_dir, video_filename, "{:010d}.jpg".format(frame)
                )
                if not os.path.exists(save_path) or overwrite:
                    cv2.imwrite(save_path, image)
                    saved_count += 1

        frame += 1

    capture.release()
    return saved_count


def video_to_frames(video_path, frames_dir, overwrite=False, every=1, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(
        video_path
    )  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    frame_chunks = [
        [i, i + chunk_size] for i in range(0, total, chunk_size)
    ]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(
        frame_chunks[-1][-1], total - 1
    )  # make sure last chunk has correct end frame, also handles case chunk_size < total

    prefix_str = "Extracting frames from {}".format(
        video_filename
    )  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(
                extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every
            )
            for f in frame_chunks
        ]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(
                i, len(frame_chunks) - 1, prefix=prefix_str, suffix="Complete"
            )  # print it's progress

    return os.path.join(
        frames_dir, video_filename
    )  # when done return the directory containing the frames


# if __name__ == "__main__":
#     # test it
#     video_to_frames(
#         video_path="data/face_finder.mp4",
#         frames_dir="/Users/mustafagumustas/screen-time/data/detected_faces",
#         overwrite=False,
#         every=1,
#         chunk_size=1000,
#     )

#     if sys.platform == "darwin":
#         multiprocessing.set_start_method("spawn")
