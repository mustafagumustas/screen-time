import numpy as np
import cv2


# face recognition, both frontal and lateral angels
weights = "models/face_detection_yunet_2022mar.onnx"
face_detector = cv2.FaceDetectorYN_create(weights, "", (1280, 720))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    _, faces = face_detector.detect(frame)
    faces = faces if faces is not None else []
    for face in faces:
        box = list(map(int, face[:4]))
        cv2.rectangle(frame, box, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
