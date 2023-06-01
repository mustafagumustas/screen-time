import cv2
import dlib
import numpy as np
from scripts.face_tools import face_degree

MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"
MAX_FRAME_COUNT = 15


def initialize_detector_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)
    return detector, predictor


def detect_faces(frame, detector):
    faces = detector(frame)
    return faces


def align_faces(frame, faces, predictor):
    aligned_faces = []
    for face in faces:
        landmarks = predictor(frame, face)
        landmarks = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        )
        aligned_face = face_degree(
            frame, landmarks
        )  # Assuming you have the face_degree function implemented
        aligned_faces.append(aligned_face)
    return aligned_faces


def crop_faces(aligned_faces):
    cropped_faces = []
    for aligned_face in aligned_faces:
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cropped_face = aligned_face[y1:y2, x1:x2]
            cropped_faces.append(cropped_face)
    return cropped_faces


def process_frames(max_frame_count):
    detector, predictor = initialize_detector_predictor()
    cap = cv2.VideoCapture(0)
    frame_count = 0  # Counter for the saved frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, detector)
        aligned_faces = align_faces(frame, faces, predictor)
        cropped_faces = crop_faces(aligned_faces)

        for cropped_face in cropped_faces:
            cv2.imshow("Cropped Face", cropped_face)
            key = cv2.waitKey(0)
            if key == ord("s"):  # Press 's' to save the current frame
                frame_count += 1
                cv2.imwrite(f"data/new_face_{frame_count}.jpg", cropped_face)
                if frame_count >= max_frame_count:
                    break
            else:
                continue

        if frame_count >= max_frame_count:
            break

        # display frame with detected faces
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


process_frames(MAX_FRAME_COUNT)
