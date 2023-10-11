import cv2
import dlib
import os
import time
import multiprocessing

predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


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

    print(f"Frames with faces extracted: {frame_count}")
    print(f"Frames with faces saved in: {output_dir}")


def face_finder(frame_path):
    image = cv2.imread(frame_path)
    rects = detector(image, 0)
    print(rects)


def non_face_remover(frame_path):
    frames = [
        os.path.join(output_directory, frame) for frame in os.listdir(output_directory)
    ]


if __name__ == "__main__":
    video_path = "data/face_finder.mp4"  # Replace with the path to your MP4 video
    output_directory = "output_faces"  # Replace with your desired output directory
    start_time = time.time()
    extract_frames(video_path, output_directory)
    # face_finder(output_directory)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
