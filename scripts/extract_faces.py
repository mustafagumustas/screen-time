import cv2
import dlib
import numpy as np
import math
from PIL import Image
from face_alignment import face_degree

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop over the detected faces
    for face in faces:
        # Get the landmarks for the face in this frame
        landmarks = predictor(gray, face)

        # Convert the landmarks to a NumPy array
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Draw the bounding box and landmarks for the detected face
        for landmark in landmarks:
            cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)

        # Rotate the frame to align the face horizontally
        rotated_frame, angle = face_degree(frame, landmarks)

        # Convert the rotated frame to grayscale
        rotated_gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the rotated grayscale frame
        faces_rotated = detector(rotated_gray)

        # Loop over the detected faces in the rotated frame
        for face_rotated in faces_rotated:
            # Get the landmarks for the rotated face
            landmarks_rotated = predictor(rotated_gray, face_rotated)

            # Convert the landmarks to a NumPy array
            landmarks_rotated = np.array(
                [[p.x, p.y] for p in landmarks_rotated.parts()]
            )

            # Draw the landmarks on the rotated frame
            for landmark in landmarks_rotated:
                cv2.circle(rotated_frame, tuple(landmark), 2, (0, 255, 0), -1)

            # Draw the bounding box for the detected face
            cv2.rectangle(
                rotated_frame,
                (face_rotated.left(), face_rotated.top()),
                (face_rotated.right(), face_rotated.bottom()),
                (0, 0, 255),
                2,
            )

        # Display the rotated frame
        cv2.imshow("Rotated Frame", rotated_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and destroy all windows
cap.release()
