import cv2
import os
import numpy as np
import time
import concurrent.futures


def extract_frame(video_path, time_in_seconds, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False

    # Calculate the frame index corresponding to the given time
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_index = int(fps * time_in_seconds)

    # Set the video capture object to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame at {time_in_seconds} seconds.")
        return False
    else:
        output_file = os.path.join(output_dir, f"frame_at_{time_in_seconds}.jpg")
        cv2.imwrite(output_file, frame)
        print(f"Frame extracted at {time_in_seconds} seconds")
        print(f"Frame saved in: {output_file}")
        return True


def extract_frames_concurrently(
    video_path, output_dir, times_in_seconds, num_concurrent
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        # Extract frames concurrently
        futures = [
            executor.submit(extract_frame, video_path, time, output_dir)
            for time in times_in_seconds
        ]

    # Ensure all tasks are completed
    concurrent.futures.wait(futures)


def extract_frames_at_times(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = frame_count / fps  # Calculate video duration in seconds

    times_in_seconds = np.arange(0, video_duration_seconds, 1)

    num_concurrent = 15  # Number of frames to extract concurrently
    extract_frames_concurrently(
        video_path, output_dir, times_in_seconds, num_concurrent
    )

    cap.release()


if __name__ == "__main__":
    video_path = "data/face_finder.mp4"  # Replace with the path to your video
    output_directory = "output_frames"  # Replace with your desired output directory
    starting_time = time.time()
    extract_frames_at_times(video_path, output_directory)
    end_time = time.time()

    print("total time duration: ", end_time - starting_time, "seconds")
