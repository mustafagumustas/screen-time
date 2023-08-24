import os
import urllib.request
import bz2
import subprocess


# Function to download and extract model
def download_and_extract_model(model_url, model_path):
    print(f"Downloading model from {model_url}")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"Model downloaded and saved to {model_path}")


# Function to extract compressed file
def extract_compressed_file(file_path, output_path):
    with open(file_path, "rb") as f:
        with open(output_path, "wb") as out:
            decompressor = bz2.BZ2Decompressor()
            for data in iter(lambda: f.read(100 * 1024), b""):
                out.write(decompressor.decompress(data))


packages_to_install = [
    "opencv-python==4.7.0",
    "numpy==1.24.3",
    "Pillow==10.0.0",
    "keras==2.11.0",
]

# Create a virtual environment
venv_name = "face_recognition_env"
subprocess.run(["python", "-m", "venv", venv_name], check=True)

# Activate the virtual environment
activate_script = os.path.join(
    venv_name, "Scripts" if os.name == "nt" else "bin", "activate"
)
subprocess.run([activate_script], shell=True, check=True)

# Install required packages with specified versions
for package in packages_to_install:
    subprocess.run(["pip", "install", package], check=True)

print("Virtual environment created and packages installed.")
print(
    f"Activate the environment using: source {activate_script}"
    if os.name != "nt"
    else f"Activate the environment using: {activate_script}"
)

# Set up folder paths
data_folder = "data"
models_folder = "models"
scripts_folder = "scripts"

# Create necessary folders if they don't exist
for folder in [data_folder, models_folder, scripts_folder]:
    os.makedirs(folder, exist_ok=True)

# Download and prepare modelsh
face_detection_model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
shape_predictor_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
face_detection_model_path = os.path.join(models_folder, "face_detection_model.onnx")
shape_predictor_compressed_path = os.path.join(
    models_folder, "shape_predictor_68_face_landmarks.dat.bz2"
)
shape_predictor_extracted_path = os.path.join(
    models_folder, "shape_predictor_68_face_landmarks.dat"
)

# Download and extract models if they don't exist
if not os.path.exists(face_detection_model_path):
    download_and_extract_model(face_detection_model_url, face_detection_model_path)

if not os.path.exists(shape_predictor_extracted_path):
    if not os.path.exists(shape_predictor_compressed_path):
        download_and_extract_model(shape_predictor_url, shape_predictor_compressed_path)
    extract_compressed_file(
        shape_predictor_compressed_path, shape_predictor_extracted_path
    )
    os.remove(shape_predictor_compressed_path)
