import cv2
import numpy as np
import math
from PIL import Image
import uuid
import os
import dlib
from keras_preprocessing.image import img_to_array
from keras.applications.resnet import preprocess_input


# landmarks locations in their list
# The mouth can be accessed through points [48, 68].
# The right eyebrow through points [17, 22].
# The left eyebrow through points [22, 27].
# The right eye using [36, 42].
# The left eye with [42, 48].
# The nose using [27, 35].
# And the jaw via [0, 17].
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


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


def save_frame(face, save_dir):
    filename = str(uuid.uuid4()) + ".jpg"
    save_path = save_dir + "/" + filename
    print(f"images saved to {save_path}")
    cv2.imwrite(save_path, face)


def resize_img(directory):
    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        # Get the full file path
        image_path = os.path.join(directory, filename)

        # Check if the file is an image (you can modify this condition based on your file types)
        if os.path.isfile(image_path) and filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            # Read the cropped face image
            cropped_face = cv2.imread(image_path)

            # Perform the preprocessing steps as mentioned earlier
            resized_face = cv2.resize(cropped_face, (224, 224))
            resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            preprocessed_face = preprocess_input(resized_face)
            return preprocessed_face
