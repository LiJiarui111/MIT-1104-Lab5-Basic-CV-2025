# MIT 1.104 Lab 5 Part B
# Basic Computer Vision & Camera Calibration

This repository contains the necessary code and instructions for the Camera Calibration Lab. This lab will help you understand the pinhole camera model and perform camera calibration using your smartphone.

## Requirements

### Software Requirements

Before starting the lab, make sure you have the following installed:

1. **Python 3.6+**
2. **OpenCV**: For computer vision functions
   ```
   pip install opencv-python
   ```
3. **NumPy**: For numerical operations
   ```
   pip install numpy
   ```
4. **Matplotlib**: For visualization
   ```
   pip install matplotlib
   ```
5. **PyYAML**: For saving calibration parameters
   ```
   pip install pyyaml
   ```
6. **tqdm**: For progress bars
   ```
   pip install tqdm
   ```

You can install all requirements at once using:
```
pip install -r requirements.txt
```

### Hardware Requirements

1. **Smartphone with a camera**
2. **Printed checkerboard pattern** (provided separately)
   - Make sure to print the pattern on a flat, rigid surface
   - Do not scale the pattern during printing
   - The checkerboard should have 9×6 internal corners (10×7 squares)

## Project Structure

```
camera_calibration_lab/
├── camera_calibration.py          # Main calibration script
├── visualize_calibration.py       # Visualization script
├── checkerboard_pattern.pdf       # Printable checkerboard pattern
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Getting Started

1. Create a folder named `calibration_images` in the project directory
2. Take approximately 15 photos of the checkerboard pattern using your smartphone
3. Transfer the images to your computer and place them in the `calibration_images` folder
4. Follow the instructions in the lab handout for running the calibration script

## Troubleshooting

### Common Issues and Solutions

#### 1. Checkerboard corners not detected

If the script cannot detect the checkerboard corners in your images, try the following:

- Ensure the entire checkerboard is visible in the image
- Make sure the checkerboard is well-lit and in focus
- Avoid reflections or shadows on the checkerboard
- Try taking the picture from a different angle

#### 2. High reprojection error

If your calibration has a high reprojection error (typically > 1 pixel):

- Use more images for calibration
- Make sure the checkerboard is completely flat
- Try to cover different parts of the image frame in your photos
- Ensure the checkerboard is correctly printed (squares should be perfect squares)

#### 3. Distorted or incorrect undistorted images

If the undistorted images look strange:

- Make sure you've specified the correct checkerboard dimensions (rows, columns, square size)
- Try with more calibration images
- Make sure your checkerboard pattern is not damaged or bent

#### 4. Import errors

If you encounter import errors:

- Make sure all required packages are installed
- Check that you're using Python 3.6 or later
- Try creating a new virtual environment and installing the dependencies again

#### 5. File not found errors

- Double-check that your image paths are correct
- Make sure the calibration_images folder is in the same directory as the scripts
- Verify that the files have .jpg, .jpeg, or .png extensions

## Camera Calibration Parameters

After running the calibration script, a `camera_calibration.yaml` file will be created with the following parameters:

- **Camera Matrix**: Contains the intrinsic parameters (focal length and principal point)
- **Distortion Coefficients**: Parameters modeling the lens distortion (radial and tangential)
- **Reprojection Error**: A measure of the calibration accuracy

These parameters can be used for various computer vision tasks, such as 3D reconstruction, augmented reality, and stereo vision.

## Understanding the Code

### camera_calibration.py

This script performs the camera calibration process:

1. Loads the calibration images
2. Detects the checkerboard corners in each image
3. Computes the camera parameters using Zhang's method
4. Calculates the reprojection error
5. Saves the calibration parameters to a YAML file
6. Generates undistorted versions of the calibration images

### visualize_calibration.py

This script visualizes the effects of calibration:

1. Loads the calibration parameters
2. Undistorts a specified image
3. Visualizes the camera's position relative to the checkerboard
4. Creates a 3D plot showing the camera and checkerboard

## Tips for Good Calibration

1. **Use at least 10-15 images** from different viewpoints
2. **Cover the entire image frame** in your collection of images
3. **Include images where the checkerboard is at an angle** to the camera
4. **Make sure the checkerboard is flat** and not bent
5. **Ensure good lighting** to get clear, high-contrast images
6. **Keep the camera settings constant** (no autofocus, no auto-exposure)
7. **The checkerboard should fill a significant portion** of the image

<!-- ## Extending the Lab

After completing the basic calibration, you can try:

1. Comparing calibration results from different smartphones
2. Measuring real-world objects using your calibrated camera
3. Creating a simple augmented reality application
4. Implementing stereo vision if you have two cameras -->

## References

1. Zhang, Z. (2000). A flexible new technique for camera calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 1330-1334.
2. OpenCV Documentation: [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)
3. Bradski, G., & Kaehler, A. (2008). Learning OpenCV: Computer vision with the OpenCV library. O'Reilly Media, Inc.
