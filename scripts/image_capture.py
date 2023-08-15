import os
import cv2
import numpy as np

# Global variables for data collection
counter = 0
max_images_per_rotation = 200
dot_size = 10
dot_x, dot_y = 0, 0
dot_angle = 0
dot_speed = 2
dot_max_radius = 50  # Adjust this value based on your preference
person_name = input("enter your name:")
# Calculate the total number of frames for one full rotation to achieve at least 200 images
num_frames_for_full_rotation = max_images_per_rotation // dot_speed

face_detector = cv2.FaceDetectorYN_create(
    "models/face_detection_yunet_2022mar.onnx", "", (0, 0)
)


def update_dot_and_capture():
    global dot_x, dot_y, dot_angle, dot_max_radius, counter

    # Update dot position for circular motion
    dot_x = width // 2 + int(dot_max_radius * np.cos(np.radians(dot_angle)))
    dot_y = height // 2 + int(dot_max_radius * np.sin(np.radians(dot_angle)))
    dot_angle += dot_speed

    # Increase dot radius over time for more angles
    dot_max_radius += 0.05

    # Capture images and save them to the appropriate folder
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    if ret:
        save_image(frame)


def save_image(frame):
    global counter

    # Save the captured image to the appropriate folder based on the counter
    person_folder = f"./data/{person_name}"
    os.makedirs(person_folder, exist_ok=True)
    image_path = os.path.join(person_folder, f"{person_name}_{counter + 1}.jpg")
    h, x, _ = frame.shape
    face_detector.setInputSize((x, h))
    _, faces = face_detector.detect(frame)
    faces = faces if faces is not None else []
    for face in faces:
        if face is not []:
            box = list(map(int, face[:4]))
            cv2.imwrite(
                image_path, frame[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
            )

    # Update the counter and check if we reached the maximum number of images for one rotation
    counter += 1
    if counter >= num_frames_for_full_rotation:
        print(f"Captured {num_frames_for_full_rotation} images in one full rotation")

    if counter >= max_images_per_rotation:
        camera.release()
        cv2.destroyAllWindows()
    else:
        cv2.putText(
            frame,
            f"Images Left: {max_images_per_rotation - counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.circle(frame, (dot_x, dot_y), dot_size, (0, 255, 0), -1)
        cv2.imshow("Data Collection for Face Recognition", frame)
        cv2.waitKey(10)  # Change this value to control dot speed


def wait_for_user():
    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Show the dot at the starting position
        start_dot_x = width // 2 + int(dot_max_radius * np.cos(np.radians(0)))
        start_dot_y = height // 2 + int(dot_max_radius * np.sin(np.radians(0)))
        cv2.circle(frame, (start_dot_x, start_dot_y), dot_size, (0, 255, 0), -1)

        cv2.putText(
            frame,
            "Press any key to start data collection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Flip the frame once at the beginning after the user presses any key
        if cv2.waitKey(1) != -1:
            break

        cv2.imshow("Data Collection for Face Recognition", frame)

    return frame


if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # Wait for the user to set up the camera angle and press any key to start data collection
    flipped_frame = wait_for_user()

    while True:
        # Display the flipped frame with the dot
        cv2.putText(
            flipped_frame,
            f"Images Left: {max_images_per_rotation - counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.circle(flipped_frame, (dot_x, dot_y), dot_size, (0, 255, 0), -1)
        cv2.imshow("Data Collection for Face Recognition", flipped_frame)

        # Capture images and update dot position
        update_dot_and_capture()

        # Break the loop if we complete at least one full rotation and capture enough images
        if counter >= max_images_per_rotation:
            break
