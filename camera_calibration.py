"""
camera_calibration.py - Camera Calibration Using OpenCV

This script performs camera calibration using multiple images of a checkerboard pattern.
It detects the checkerboard corners in each image, calculates the camera parameters,
and saves the calibration results to a YAML file.

Usage:
    python camera_calibration.py --dir <images_directory> --rows <rows> --cols <cols> --square_size <square_size>

Arguments:
    --dir: Path to the directory containing the calibration images
    --rows: Number of internal corners in the checkerboard pattern (rows)
    --cols: Number of internal corners in the checkerboard pattern (columns)
    --square_size: Physical size of each checkerboard square in meters

Example:
    python camera_calibration.py --dir ./calibration_images/ --rows 9 --cols 6 --square_size 0.023
"""

import numpy as np
import cv2
import glob
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Camera calibration using checkerboard pattern')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing calibration images')
    parser.add_argument('--rows', type=int, required=True, help='Number of internal corners in rows')
    parser.add_argument('--cols', type=int, required=True, help='Number of internal corners in columns')
    parser.add_argument('--square_size', type=float, required=True, help='Size of each checkerboard square in meters')
    parser.add_argument('--output', type=str, default='camera_calibration.yaml', help='Output file for calibration parameters')
    parser.add_argument('--visualize', action='store_true', help='Visualize corner detection and undistortion')
    return parser.parse_args()

def calibrate_camera(images_dir, rows, cols, square_size, output_file, visualize=False):
    """
    Calibrate the camera using a set of checkerboard images.
    
    Args:
        images_dir (str): Directory containing the calibration images
        rows (int): Number of internal corners in rows
        cols (int): Number of internal corners in columns
        square_size (float): Size of each checkerboard square in meters
        output_file (str): Path to save the calibration results
        visualize (bool): Whether to visualize the results
        
    Returns:
        tuple: Camera matrix, distortion coefficients, rotation vectors, translation vectors, and reprojection error
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...., (cols-1,rows-1,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get all image paths
    images = glob.glob(os.path.join(images_dir, '*.jpg')) + \
             glob.glob(os.path.join(images_dir, '*.png')) + \
             glob.glob(os.path.join(images_dir, '*.jpeg'))
    
    if not images:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Found {len(images)} images. Processing...")
    
    # Create a directory for storing calibration results
    output_dir = 'calibration_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directory for undistorted images
    undistorted_dir = os.path.join(output_dir, 'undistorted')
    os.makedirs(undistorted_dir, exist_ok=True)
    
    # Process each image
    successful_images = 0
    
    for img_path in tqdm(images):
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        
        # If corners are found, add object points and image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            successful_images += 1
            
            # Draw and display the corners if visualize is True
            if visualize:
                img_corners = img.copy()
                cv2.drawChessboardCorners(img_corners, (cols, rows), corners2, ret)
                cv2.imshow('Chessboard Corners', img_corners)
                cv2.waitKey(500)  # Display for 500ms
        else:
            print(f"Could not find chessboard corners in {img_path}")
    
    if visualize:
        cv2.destroyAllWindows()
    
    if successful_images < 3:
        raise ValueError(f"Found corners in only {successful_images} images. At least 3 are required.")
    
    print(f"Successfully processed {successful_images} out of {len(images)} images.")
    
    # Calibrate the camera
    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Mean reprojection error: {mean_error/len(objpoints)}")
    
    # Save the calibration parameters
    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'distortion_coefficients': dist.tolist(),
        'reprojection_error': float(mean_error/len(objpoints)),
        'image_width': gray.shape[1],
        'image_height': gray.shape[0],
        'calibration_time': {
            'successful_images': successful_images,
            'total_images': len(images)
        }
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f)
    
    print(f"Calibration parameters saved to {output_file}")
    
    # Undistort and save a few sample images
    print("Generating undistorted images...")
    for i, img_path in enumerate(images[:5]):  # Process the first 5 images as samples
        if i >= successful_images:
            break
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get the new camera matrix
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Undistort the image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save the undistorted image
        base_name = os.path.basename(img_path)
        undistorted_path = os.path.join(undistorted_dir, f"undistorted_{base_name}")
        cv2.imwrite(undistorted_path, dst)
        
        # Display the original and undistorted images side by side
        if visualize:
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
            plt.title('Undistorted Image')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{base_name}"))
            plt.close()
    
    return mtx, dist, rvecs, tvecs, mean_error/len(objpoints)

def main():
    """Main function."""
    args = parse_args()
    
    # Run the calibration
    calibrate_camera(
        args.dir,
        args.rows,
        args.cols,
        args.square_size,
        args.output,
        args.visualize
    )

if __name__ == "__main__":
    main()
