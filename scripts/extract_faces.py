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

# Create window
cv2.namedWindow("Face detection and alignment", cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Display the original frame if no faces are detected
    if len(faces) == 0:
        cv2.imshow("Face detection and alignment", frame)
    else:
        # Get the first face in the frame
        face = faces[0]

        # Get the landmarks for the face in this frame
        landmarks = predictor(gray, face)

        # Convert the landmarks to a NumPy array
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Draw the bounding box and landmarks for the detected face
        for landmark in landmarks:
            cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)

        # Calculate the rotation degree and rotate the image accouring to that
        rotated_frame = face_degree(frame, landmarks)

        # Zoom into the face and display it in the window
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        zoomed_face = rotated_frame[y : y + h, x : x + w]
        zoomed_face = cv2.resize(zoomed_face, (500, 500))
        cv2.imshow("Face detection and alignment", zoomed_face)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and destroy the window
cap.release()
cv2.destroyAllWindows()
