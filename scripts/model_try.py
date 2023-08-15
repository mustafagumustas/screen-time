import os
import cv2
import dlib
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input


# Function to get class names
def get_class_names(dir_path):
    return sorted(
        [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        + ["unknown"]
    )


# Directory containing your dat
data_dir = "data"  # change to the path of your data directory

# Load the trained model
model = load_model("face_recognition_model.h5")

# Create a list of class names
class_names = get_class_names(data_dir)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Set a confidence threshold
confidence_threshold = 0.75

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    rects = detector(gray, 0)

    # Iterate over all detected faces
    for rect in rects:
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()

        # Extract face
        face = frame[y : y + h, x : x + w]

        # Preprocess the face image for MobileNetV2
        image = cv2.resize(face, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Use the model to predict the image class
        preds = model.predict(image)
        max_prob = np.max(preds[0])

        if max_prob > confidence_threshold:
            label_index = np.argmax(
                preds[0]
            )  # get the index of the label with the highest probability
            # Get the class name and confidence score
            label = class_names[label_index]
            confidence = max_prob
        else:
            label = "unknown"
            confidence = (
                max_prob  # You can set the confidence score for "unknown" class
            )

        # Draw a rectangle around the face and add the label and confidence score
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = f"{label}{confidence:.2f}"
        cv2.putText(
            frame,
            label_text,
            (x, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # If 'q' is pressed on the keyboard, exit this loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Close the video feed
cap.release()
cv2.destroyAllWindows()
