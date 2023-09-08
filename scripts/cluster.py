import os
import numpy as np
import cv2

source = "data/face_finder.mp4"
output_folder = "data/detected_faces"  # Specify the folder to save detected faces

# Set the confidence threshold for face detection
confidence_threshold = 0.99  # Adjust this threshold as needed

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def main():
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        exit()

    weights = "models/face_detection_yunet_2022mar.onnx"
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

    frame_count = 0

    while True:
        result, image = capture.read()
        image = cv2.flip(image, 1)
        if result is False:
            cv2.waitKey(0)
            break

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            box = list(map(int, face[:4]))
            confidence = face[-1]  # Extract the confidence score

            if confidence >= confidence_threshold:
                # Save the frame with the detected face
                frame_count += 1
                frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, image)

                color = (0, 0, 255)
                thickness = 2

                # Display the confidence score on the image
                text = f"Confidence: {confidence:.2f}"
                cv2.putText(
                    image,
                    text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

        cv2.imshow("face detection", image)

        # Adjust the frame rate here
        key = cv2.waitKey(1)  # Display each frame for 100 milliseconds

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
