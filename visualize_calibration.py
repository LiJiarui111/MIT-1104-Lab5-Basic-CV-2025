"""
visualize_calibration.py - Visualize Camera Calibration Results

This script visualizes the effects of camera calibration by undistorting an image
and showing the camera's position relative to the checkerboard.

Usage:
    python visualize_calibration.py --calib_file <calibration_file> --image_path <image_path> 
                                   [--rows <rows> --cols <cols> --square_size <square_size>]

Arguments:
    --calib_file: Path to the calibration YAML file
    --image_path: Path to an image to undistort
    --rows: Number of internal corners in the checkerboard pattern (rows)
    --cols: Number of internal corners in the checkerboard pattern (columns)
    --square_size: Physical size of each checkerboard square in meters

Example:
    python visualize_calibration.py --calib_file camera_calibration.yaml --image_path ./calibration_images/img1.jpg --rows 9 --cols 6 --square_size 0.023
"""

import numpy as np
import cv2
import argparse
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize camera calibration results')
    parser.add_argument('--calib_file', type=str, required=True, help='Path to calibration YAML file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to an image to undistort')
    parser.add_argument('--rows', type=int, default=7, help='Number of internal corners in rows')
    parser.add_argument('--cols', type=int, default=5, help='Number of internal corners in columns')
    parser.add_argument('--square_size', type=float, default=0.023, help='Size of each checkerboard square in meters')
    return parser.parse_args()

def load_calibration_params(calib_file):
    """Load camera calibration parameters from a YAML file."""
    with open(calib_file, 'r') as f:
        calib_data = yaml.load(f, Loader=yaml.SafeLoader)
    
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['distortion_coefficients'])
    
    return camera_matrix, dist_coeffs

def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistort an image using camera calibration parameters."""
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted, newcameramtx

def draw_camera_position(image, object_points, camera_matrix, dist_coeffs, rvec, tvec):
    """Draw camera position and coordinate axis on the image."""
    # Define coordinate system axis points
    axis = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
    
    # Project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    
    # Draw coordinate axis
    img = image.copy()
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 5)  # X-axis (red)
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 5)  # Y-axis (green)
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255,0,0), 5)  # Z-axis (blue)
    
    return img

# tvec: translation vector, the position of the camera in the world coordinate system
# rvec: rotation vector, the rotation of the camera in the world coordinate system
def plot_camera_checkerboard(rvec, tvec, camera_matrix, object_points):
    """Plot the camera and checkerboard in 3D space correctly showing camera above checkerboard."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Ensure tvec is in the right shape
    tvec = tvec.reshape(3)
    
    # In OpenCV, the checkerboard is at Z=0 and the camera looks toward -Z
    # We need to transform this to a more intuitive visualization where Z is up
    
    # Create world points for checkerboard (Z=0 plane)
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], 
               c='b', marker='.', s=20, label='Checkerboard Corners')
    
    # Calculate camera position in world coordinates
    # The extrinsic parameters define transform from world to camera
    # So camera center = -R^T * t
    camera_center = -np.dot(R.T, tvec)
    
    # Plot camera center
    ax.scatter(camera_center[0], camera_center[1], camera_center[2], 
               c='r', marker='o', s=100, label='Camera Center')
    
    # Scale for camera axes visualization
    scale = 0.05  
    
    # Camera axes in world coordinates (columns of R.T)
    # In camera coordinates: x right, y down, z forward
    # We want to show this intuitively
    x_axis = camera_center + scale * R.T[:, 0]  # Camera's x-axis (red, right)
    y_axis = camera_center + scale * R.T[:, 1]  # Camera's y-axis (green, down) 
    z_axis = camera_center + scale * R.T[:, 2]  # Camera's z-axis (blue, forward)
    
    # Draw camera axes
    ax.plot([camera_center[0], x_axis[0]], 
            [camera_center[1], x_axis[1]], 
            [camera_center[2], x_axis[2]], 'r-', linewidth=2, label='X axis')
    
    ax.plot([camera_center[0], y_axis[0]], 
            [camera_center[1], y_axis[1]], 
            [camera_center[2], y_axis[2]], 'g-', linewidth=2, label='Y axis')
    
    ax.plot([camera_center[0], z_axis[0]], 
            [camera_center[1], z_axis[1]], 
            [camera_center[2], z_axis[2]], 'b-', linewidth=2, label='Z axis')
    
    # Draw a line from camera to checkerboard center to show viewing direction
    checkerboard_center = np.mean(object_points, axis=0)
    ax.plot([camera_center[0], checkerboard_center[0]], 
            [camera_center[1], checkerboard_center[1]], 
            [camera_center[2], checkerboard_center[2]], 
            'k--', alpha=0.7, linewidth=1, label='View Direction')
    
    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Camera and Checkerboard in 3D Space')
    
    # Add a small flat grid at z=0 to represent the checkerboard plane
    min_x, max_x = np.min(object_points[:, 0]), np.max(object_points[:, 0])
    min_y, max_y = np.min(object_points[:, 1]), np.max(object_points[:, 1])
    
    pad = 0.08  # Some padding around the checkerboard
    
    xx, yy = np.meshgrid(np.linspace(min_x-pad, max_x+pad, 10),
                         np.linspace(min_y-pad, max_y+pad, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # Set a good viewing angle
    ax.view_init(elev=45, azim=-60)
    
    # Make the visualization more intuitive by controlling the aspect ratio
    # Make sure camera is visibly above the checkerboard
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_min = min(camera_center[2], 0)
    z_max = max(camera_center[2], 0) 
    z_range = z_max - z_min
    
    # Ensure Z range is at least 30% of max of X and Y range
    # This helps show the camera height more clearly
    desired_z_range = max(x_range, y_range) * 0.5
    if z_range < desired_z_range:
        z_mid = (z_min + z_max) / 2
        z_min = z_mid - desired_z_range / 2
        z_max = z_mid + desired_z_range / 2
    
    # Set axis limits
    ax.set_xlim(min_x-pad, max_x+pad)
    ax.set_ylim(min_y-pad, max_y+pad)
    ax.set_zlim(z_min-pad, z_max+pad*2)  # Extra padding on top for visibility
    
    # Handle legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    return fig

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Load calibration parameters
        camera_matrix, dist_coeffs = load_calibration_params(args.calib_file)
        
        # Read the image
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {args.image_path}")
        
        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Undistort the image
        undistorted, newcameramtx = undistort_image(image, camera_matrix, dist_coeffs)
        
        # Try to find checkerboard corners for visualization
        ret, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
        
        # Prepare figure for visualization
        plt.figure(figsize=(15, 10))
        
        # Show original and undistorted images
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        plt.title('Undistorted Image')
        plt.axis('off')
        
        # If corners were found, visualize camera position
        if ret:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw and display the corners
            img_corners = image.copy()
            cv2.drawChessboardCorners(img_corners, (args.cols, args.rows), corners2, ret)
            plt.subplot(223)
            plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
            plt.title('Detected Corners')
            plt.axis('off')
            
            # Create object points
            objp = np.zeros((args.rows * args.cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_size
            # Invert the X axis (this will naturally invert the Z axis as well)
            objp[:, 0] = -objp[:, 0]
            
            # Find the rotation and translation vectors
            # The translation vector is the position of the camera in the world coordinate system
            # parameters: 
            # objp: object points in the world coordinate system
            # corners2: detected corners in the image plane, the 2D coordinates of the corners in the image
            # camera_matrix: camera matrix, obtained from the calibration
            # dist_coeffs: distortion coefficients, obtained from the calibration
            # rvecs: rotation vector, the rotation of the camera in the world coordinate system
            # tvecs: translation vector, the position of the camera in the world coordinate system
            # This function returns the extrinsic parameters: rotation and translation of the camera in the world coordinate system
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, dist_coeffs)
            
            # Draw camera position
            img_pose = draw_camera_position(image, objp, camera_matrix, dist_coeffs, rvecs, tvecs)
            plt.subplot(224)
            plt.imshow(cv2.cvtColor(img_pose, cv2.COLOR_BGR2RGB))
            plt.title('Camera Position')
            plt.axis('off')
            
            # Create output directory if it doesn't exist
            output_dir = 'visualization_results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the visualization
            plt.tight_layout()
            base_name = os.path.basename(args.image_path)
            plt.savefig(os.path.join(output_dir, f"visualization_{base_name}"))
            
            # Plot 3D visualization of camera and checkerboard
            fig = plot_camera_checkerboard(rvecs, tvecs, camera_matrix, objp)
            fig.savefig(os.path.join(output_dir, f"3d_visualization_{base_name}"))
            
            print(f"Visualization saved to {output_dir}")
            plt.show()
        else:
            print("Could not find checkerboard corners in the provided image.")
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()
