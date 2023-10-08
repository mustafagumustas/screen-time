import cv2
import os
import numpy as np
import time


def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % fps == 0:
            output_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_file, frame)

        frame_count += 1

    cap.release()

    print(f"Frames extracted: {frame_count}")
    print(f"Frames saved in: {output_dir}")


def main():
    video_path = "data/face_finder.mp4"  # Replace with the path to your video
    output_directory = "output_faces"  # Replace with your desired output directory

    start_time = time.time()

    extract_frames(video_path, output_directory)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()
