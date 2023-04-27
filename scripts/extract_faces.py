import cv2
import dlib
import numpy as np
import math
from PIL import Image
from face_tools import face_degree, save_frame

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector(frame)

    for face in faces:
        landmarks = predictor(frame, face)
        landmarks = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        )
        aligned_face = face_degree(frame, landmarks)

        # detect face on aligned frame
        aligned_gray = cv2.cvtColor(aligned_face, cv2.IMREAD_COLOR)
        aligned_faces = detector(aligned_face)

        for aligned_face in aligned_faces:
            x1 = aligned_face.left()
            y1 = aligned_face.top()
            x2 = aligned_face.right()
            y2 = aligned_face.bottom()

            # draw rectangle around face
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # crop face from aligned frame
            cropped_face = aligned_gray[y1:y2, x1:x2]

            # display cropped face
            cv2.imshow("Cropped Face", cropped_face)
            save_frame(cropped_face, "data/new_face")

    # display frame with detected faces
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
