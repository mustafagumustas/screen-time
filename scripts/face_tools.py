import cv2
import numpy as np
import math
from PIL import Image
import uuid
import dlib

# landmarks locations in their list
# The mouth can be accessed through points [48, 68].
# The right eyebrow through points [17, 22].
# The left eyebrow through points [22, 27].
# The right eye using [36, 42].
# The left eye with [42, 48].
# The nose using [27, 35].
# And the jaw via [0, 17].

predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def face_degree(frame, landmarks):
    # because we flip the frame left and right also flipped
    right_eye = landmarks[42:48]
    left_eye = landmarks[36:42]
    nose = landmarks[27:35]

    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)
    nose_center = np.mean(nose, axis=0).astype(int)

    # find rotation direction
    if left_eye_center[1] > right_eye_center[1]:
        point_3rd = right_eye_center[0], left_eye_center[1]
        direction = -1  # rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = left_eye_center[0], right_eye_center[1]
        direction = 1  # rotate inverse direction of clock
        print("rotate to inverse clock direction")

    # draw a triangle between eyes and nose centers
    # cv2.line(
    #     frame,
    #     left_eye_center,
    #     right_eye_center,
    #     (0, 255, 0),
    #     thickness=3,
    #     lineType=8,
    # )
    # cv2.line(frame, left_eye_center, nose_center, (0, 255, 0), thickness=3, lineType=8)
    # cv2.line(frame, right_eye_center, nose_center, (0, 255, 0), thickness=3, lineType=8)
    # cv2.circle(frame, point_3rd, 2, (255, 0, 0), 2)
    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, point_3rd)
    c = euclidean_distance(right_eye_center, left_eye_center)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    # print("cos(a) = ", cos_a)
    angle = np.arccos(cos_a)
    # print("angle: ", angle," in radian")

    angle = (angle * 180) / math.pi
    print("angle: ", angle, " in degree")

    if direction == -1:
        angle = 90 - angle

    print("angle: ", angle, " in degree")

    # rotate image
    new_img = Image.fromarray(frame)
    new_img = np.array(new_img.rotate(direction * angle))
    # cv2.imshow("Face Detection", new_img)
    return new_img


def save_face(face, save_dir):
    filename = str(uuid.uuid4()) + ".jpg"
    save_path = save_dir + "/" + filename
    cv2.imwrite(save_path, face)


def face_detector(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)
    # Display the original frame if no faces are detected
    if len(faces) == 0:
        # cv2.imshow("Face detection and alignment", frame)
        pass
    else:
        # Get the first face in the frame
        face = faces[0]
        # Zoom into the face and display it in the window
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # Get the landmarks for the face in this frame
        landmarks = predictor(gray, face)


def face_degree_YuNet(frame, landmarks):
    # because we flip the frame left and right also flipped
    right_eye = landmarks[4:6]
    left_eye = landmarks[6:8]
    nose = landmarks[8:10]
    print(right_eye, left_eye, nose)
    # find rotation direction
    if left_eye[1] > right_eye[1]:
        point_3rd = right_eye[0], left_eye[1]
        direction = -1  # rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = left_eye[0], right_eye[1]
        direction = 1  # rotate inverse direction of clock
        print("rotate to inverse clock direction")

    # draw a triangle between eyes and nose centers
    # cv2.line(
    #     frame,
    #     left_eye,
    #     right_eye,
    #     (0, 255, 0),
    #     thickness=3,
    #     lineType=8,
    # )
    # cv2.line(frame, left_eye, nose, (0, 255, 0), thickness=3, lineType=8)
    # cv2.line(frame, right_eye, nose, (0, 255, 0), thickness=3, lineType=8)
    # cv2.circle(frame, point_3rd, 2, (255, 0, 0), 2)
    a = euclidean_distance(left_eye, point_3rd)
    b = euclidean_distance(right_eye, point_3rd)
    c = euclidean_distance(right_eye, left_eye)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    # print("cos(a) = ", cos_a)
    angle = np.arccos(cos_a)
    # print("angle: ", angle," in radian")

    angle = (angle * 180) / math.pi
    print("angle: ", angle, " in degree")

    if direction == -1:
        angle = 90 - angle

    print("angle: ", angle, " in degree")

    # rotate image
    new_img = Image.fromarray(frame)
    new_img = np.array(new_img.rotate(direction * angle))
    # cv2.imshow("Face Detection", new_img)
    return new_img
