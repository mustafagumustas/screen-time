# Face Recognition and Screen Time Calculation

This Python program takes video or webcam footage as input, recognizes faces, assigns unique ID numbers to each person in every frame, and calculates their total screen time.

## Requirements

- Python 3
- OpenCV
- TensorFlow
- dlib
- NumPy
- Scikit-learn (optional, if you plan to use machine learning algorithms for face clustering or classification)

## Installation

First, make sure Python 3 is installed on your machine. Then, install the required Python libraries using pip:

```shell
pip install opencv-python tensorflow dlib numpy scikit-learn
```
## Usage

The program will display the output in real time, showing the detected faces, their assigned ID numbers, and their screen time.

To train the model, you will need to provide at least 15 images of each person that the system should recognize. The face of each person should be visible in the images. Put these images in the data folder, in a subfolder named after each person. For example:
/data
    /mustafa
        img1.jpg
        img2.jpg
    /your_name
        img1.jpg
        img2.jpg

After preparing the images, simply run main.py:

```shell
python main.py
```

This will create a model with a .h5 extension.

Next, run model_try.py to test the system:
```shell
python model_try.py
```
This will open the camera and you should see that the system recognizes your face and labels it with the correct name.

## Image Augmentation

To improve the model's accuracy, the system uses image augmentation techniques. The augment_images function within the data_preprocessing.py script applies random transformations to the images. This includes flipping the images along both axes, random brightness and contrast adjustments, random saturation and hue changes.

Note that these transformations are performed using TensorFlow's image operations. The augmented images are then converted back to numpy arrays for compatibility with other parts of the program.


## Acknowledgments

This project was inspired by the OpenCV documentation and the work of various face recognition and tracking researchers.

If you use this program in your research or project, please consider citing this repository as a reference.