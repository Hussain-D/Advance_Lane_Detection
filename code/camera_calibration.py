"""
Camera Calibration module
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CameraCalibration:
    """
    Class to contain the calibration parameters
    and code
    """
    def __init__(self, nx = 8, ny = 6, calibration_dir = './../camera_cal',
                 output_dir = './../output_undistort'):
        """
        nx: Number of corners in x
        ny: Number of corners in y
        calibration_dir: Directory containing images for calibration
        """
        self.nx = nx
        self.ny = ny
        self.img_dir = calibration_dir

        self.output_dir = output_dir

    def calibrate(self, visualize = False):
        """
        Function to start the calibration process
        """
        # Arrays to store object_points and image_points
        object_points = []
        image_points = []

        # Generate default object point
        object_point = np.zeros((self.nx * self.ny, 3), np.float32)
        object_point[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2) 

        # Get all image and object points
        if visualize == False:
            image_files = os.listdir(self.img_dir)
            for image in image_files:
                ret, corners = self._determine_corners(self.img_dir + "/" + image)

                if ret == True:
                    image_points.append(corners)
                    object_points.append(object_point)

            # Determine the camera matrix
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, 
                                        (1280, 720), None, None)

            self.camera_matrix = mtx
            self.distortion_coeff = dist

        else:
            image_file = os.listdir(self.img_dir)[1]
            ret, corners = self._determine_corners(self.img_dir + "/" + image_file)

            image = cv2.imread(self.img_dir + "/" + image_file)
            image = cv2.drawChessboardCorners(image, (self.nx, self.ny), corners, ret)
            plt.imshow(image)
            plt.show()

    def undistort(self, image_file, output_file):
        """
        Function to undistort and show a given image (file location)
        """
        image = cv2.imread(image_file)
        undist = cv2.undistort(image, self.camera_matrix, self.distortion_coeff, 
                 None, self.camera_matrix)

        cv2.imwrite(output_file, undist)
        # plt.imshow(undist)
        # plt.show()

    def undistort_image(self, image):
        """
        Function to undistort a given image
        """
        undist = cv2.undistort(image, self.camera_matrix, self.distortion_coeff,
                 None, self.camera_matrix)

        return undist

    def _determine_corners(self, image):
        """
        Private function to determine corners
        
        Parameters
        ---
        image: image file to determine corners for
        """
        # Read image
        img = cv2.imread(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

        return ret, corners

# if __name__ == "__main__":
#     calibration = CameraCalibration(9, 6)
#     calibration.calibrate()
#
#     print("--- Calibration Matrix ---")
#     print(calibration.camera_matrix)
#     print("--- Distortion Coefficients ---")
#     print(calibration.distortion_coeff)
#
#     # calibration.undistort("camera_cal/calibration2.jpg")
#     print("Converting test images")
#     test_images = os.listdir("./../test_images/")
#     for image in test_images:
#         print(f"Converted image {image}")
#         image_file = f"./../test_images/{image}"
#         output_file = f"./../output_undistort/{image}"
#         calibration.undistort(image_file, output_file)
