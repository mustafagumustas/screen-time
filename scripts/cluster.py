import cv2
import dlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys
import numpy as np

face_counter = 0


def print_progress(iteration, total, prefix="", suffix="", decimals=3, bar_length=100):
    # Function to create a progress bar
    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),
    sys.stdout.flush()


def detect_and_save_faces(frame, output_dir):
    # Load the pre-trained face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Detect faces in the frame
    faces = detector(frame)

    for _, face in enumerate(faces):
        # Get facial landmarks
        landmarks = predictor(frame, face)

        # Extract the coordinates of the facial landmarks
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()

        # Crop the face from the frame using the facial landmarks
        face_image = frame[top:bottom, left:right]

        # Save the cropped face as an image in the output directory with a unique name
        # output_filename = os.path.join(output_dir, f"face_{face_counter}.jpg")
        base, extension = os.path.splitext(output_dir)
        cv2.imwrite(f"{base}_{_}{extension}", face_image)

        # face_counter += 1  # Increment the counter for the next face


# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

source = "data/face_finder.mp4"
output_dir = "output_faces"  # Directory to save the detected faces

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


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

    while frame < end:  # lets loop through the frames until the end
        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if (
            image is None
        ):  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if (
            frame % every == 0
        ):  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(
                frames_dir, video_filename, "{:010d}.jpg".format(frame)
            )  # create the save path
            if (
                not os.path.exists(save_path) or overwrite
            ):  # if it doesn't exist or we want to overwrite anyways
                detect_and_save_faces(image, save_path)
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


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


def generate_face_embeddings(face_image):
    # Generate the face embedding for a given face image
    face_encoding = face_rec_model.compute_face_descriptor(face_image)
    return np.array(face_encoding)


if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn")

    video_to_frames(
        video_path="data/bolum1.mp4",
        frames_dir="datatest_frames",
        overwrite=False,
        every=1,
        chunk_size=1000,
    )

    # Load the pre-trained face recognition model (dlib's model)
    face_rec_model = dlib.face_recognition_model_v1(
        "models/dlib_face_recognition_resnet_model_v1.dat"
    )
