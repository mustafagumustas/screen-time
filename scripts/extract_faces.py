import cv2
import dlib
import numpy as np

# landmarks locations in their list
# The mouth can be accessed through points [48, 68].
# The right eyebrow through points [17, 22].
# The left eyebrow through points [22, 27].
# The right eye using [36, 42].
# The left eye with [42, 48].
# The nose using [27, 35].
# And the jaw via [0, 17].

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)

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

        # because we flip the frame left and right also flipped
        right_eye = landmarks[42:48]
        left_eye = landmarks[36:42]
        nose = landmarks[27:35]

        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)
        nose_center = np.mean(nose, axis=0).astype(int)

        # ----------------------
        # find rotation direction
        if left_eye_center[1] > right_eye_center[1]:
            point_3rd = left_eye_center[0], right_eye_center[1]
            direction = -1  # rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = right_eye_center[1], left_eye_center[1]
            direction = 1  # rotate inverse direction of clock
            print("rotate to inverse clock direction")

        # ----------------------

        # cv2.circle(frame, point_3rd, 2, (255, 0, 0), 2)
        # Draw the bounding box and landmarks for the detected face
        cv2.rectangle(
            frame,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (0, 0, 255),
            2,
        )
        # cv2.line(frame, right_eye_center, left_eye_center, (67, 67, 67), 1)
        # cv2.line(frame, left_eye_center, point_3rd, (67, 67, 67), 1)
        # cv2.line(frame, right_eye_center, point_3rd, (67, 67, 67), 1)
        # draw a triangle between eyes and nose centers

        cv2.line(
            frame,
            left_eye_center,
            right_eye_center,
            (0, 255, 0),
            thickness=3,
            lineType=8,
        )
        cv2.line(
            frame, left_eye_center, nose_center, (0, 255, 0), thickness=3, lineType=8
        )
        cv2.line(
            frame, right_eye_center, nose_center, (0, 255, 0), thickness=3, lineType=8
        )
        # for landmark in landmarks:
        #     cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)

    # Display the frame with bounding boxes and landmarks
    cv2.imshow("Face Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
