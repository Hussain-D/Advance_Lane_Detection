"""
Transformation Functions after undistortion
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class ImageTransformation:
    """
    Module for image transformation pipeline

    - Convert to Binary Image
    - Perspective Transform
    """
    def __init__(self):
        pass

    def binary_pipeline(self, image_file, output_file, 
        threshold_1, threshold_2, threshold_3):
        """
        Function to make image go through pipeline
        """
        image = cv2.imread(image_file)

        # Convert to 3 channels
        channel_1, channel_2, channel_3 = self._split_channels(image, cv2.COLOR_BGR2HLS)
        channel_4, channel_5, channel_6 = self._split_channels(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Process the different channels
        final_1 = self._process_gradient(gray, threshold_1)
        final_2 = self._process_threshold(channel_4, threshold_2)
        final_3 = self._process_threshold(channel_3, threshold_3)

        color_binary = np.dstack((final_1, final_2, final_3)) * 255

        combined_binary = np.zeros_like(channel_1)
        combined_binary[(final_1 == 1) | (final_2 == 1) | (final_3 == 1)] = 1

        # plt.imshow(combined_binary)
        # plt.show()

        cv2.imwrite(output_file, color_binary)

    def binary_pipeline_image(self, image, threshold_1, threshold_2, threshold_3):
        """
        Function to do channel based image transformations
        """
        # Convert to 3 channels
        channel_1, channel_2, channel_3 = self._split_channels(image, cv2.COLOR_BGR2HLS)
        channel_4, channel_5, channel_6 = self._split_channels(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Process the different channels
        final_1 = self._process_gradient(gray, threshold_1)
        final_2 = self._process_threshold(channel_4, threshold_2)
        final_3 = self._process_threshold(channel_3, threshold_3)

        color_binary = np.dstack((final_1, final_2, final_3)) * 255

        combined_binary = np.zeros_like(channel_1)
        combined_binary[(final_1 == 1) | (final_2 == 1) | (final_3 == 1)] = 1

        return color_binary

    def transform_pipeline(self, image_file, output_file):
        """
        Function for perspective transformation pipeline
        """
        image = cv2.imread(image_file)
        image_shape = image.shape[::-1]

        #Used homography matrix for image warping
        #Area of projection
        dest_points = np.float32([
            [image_shape[1] / 5, 0],
            [image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, 0]
        ])

        #Area of interest
        src_points = np.float32([
            [585, 460],
            [203, 720],
            [1127, 720],
            [695, 460]
        ])

        transform_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
        warped = cv2.warpPerspective(image, transform_matrix, 
                 (image_shape[1], image_shape[2]), flags=cv2.INTER_LINEAR)

        # plt.imshow(image)
        # plt.show()
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        binary_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

        cv2.imwrite(output_file, binary_image)

    def transform_pipeline_rgb(self, image_file, output_file):
        """
        Function for perspective transformation pipeline
        RGB image
        """
        image = cv2.imread(image_file)
        image_shape = image.shape[::-1]

        dest_points = np.float32([
            [image_shape[1] / 5, 0],
            [image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, 0]
        ])

        src_points = np.float32([
            [585, 460],
            [203, 720],
            [1127, 720],
            [695, 460]
        ])

        transform_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
        warped = cv2.warpPerspective(image, transform_matrix, 
                 (image_shape[1], image_shape[2]), flags=cv2.INTER_LINEAR)

        # plt.imshow(image)
        # plt.show()

        cv2.imwrite(output_file, warped)

    def transform_pipeline_image(self, image):
        """
        Function for perspective transform on image
        """
        image_shape = image.shape[::-1]

        dest_points = np.float32([
            [image_shape[1] / 5, 0],
            [image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, 0]
        ])

        src_points = np.float32([
            [585, 460],
            [203, 720],
            [1127, 720],
            [695, 460]
        ])

        transform_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
        warped = cv2.warpPerspective(image, transform_matrix, 
                 (image_shape[1], image_shape[2]), flags=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        binary_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

        return binary_image

    def untransform_pipeline(self, image_file, original_image_file, output_file):
        """
        Function to undo the perspective transform of the image
        """
        image = cv2.imread(image_file)
        original_image = cv2.imread(original_image_file)
        image_shape = original_image.shape[::-1]

        dest_points = np.float32([
            [image_shape[1] / 5, 0],
            [image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, 0]
        ])

        src_points = np.float32([
            [585, 460],
            [203, 720],
            [1127, 720],
            [695, 460]
        ])

        inverse_transform_matrix = cv2.getPerspectiveTransform(dest_points, src_points)
        warped = cv2.warpPerspective(image, inverse_transform_matrix, 
                 (image_shape[1], image_shape[2]), flags=cv2.INTER_LINEAR)

        output_image = cv2.addWeighted(original_image, 0.8, warped, 1.0, 0)
        cv2.imwrite(output_file, output_image)

    def untransform_pipeline_image(self, image, original_image):
        """
        Function untransform a perspective transform
        """
        image_shape = original_image.shape[::-1]

        dest_points = np.float32([
            [image_shape[1] / 5, 0],
            [image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, image_shape[2]],
            [4 * image_shape[1] / 5, 0]
        ])

        src_points = np.float32([
            [585, 460],
            [203, 720],
            [1127, 720],
            [695, 460]
        ])

        inverse_transform_matrix = cv2.getPerspectiveTransform(dest_points, src_points)
        warped = cv2.warpPerspective(image, inverse_transform_matrix, 
                 (image_shape[1], image_shape[2]), flags=cv2.INTER_LINEAR)

        output_image = cv2.addWeighted(original_image, 0.8, warped, 1.0, 0)

        return output_image

    def _process_threshold(self, image, threshold):
        """
        Private function to do magnitude thresholding
        """
        binary = np.zeros_like(image)
        binary[(image >= threshold[0]) & (image <= threshold[1])] = 1

        return binary

    def _process_gradient(self, image, threshold):
        """
        Private function to do gradient thresholding
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

        return sx_binary

    def _split_channels(self, image, color_flag):
        """
        Private function to split channels
        """
        channel_image = cv2.cvtColor(image, color_flag)

        channel_1 = channel_image[:,:,0]
        channel_2 = channel_image[:,:,1]
        channel_3 = channel_image[:,:,2]

        return channel_1, channel_2, channel_3

# if __name__ == "__main__":
#     transformation = ImageTransformation()
#     operation = "untransform"
#
#     if operation == "transform":
#         image_files = os.listdir("./../output_undistort")
#         for image_file in image_files:
#             print("Transforming " + image_file)
#             transformation.binary_pipeline(
#                 "./../output_undistort/" + image_file,
#                 "./../output_binary/" + image_file,
#                 (40, 60), (220, 255), (90, 100)
#                 )
#
#             transformation.transform_pipeline(
#                 "./../output_binary/" + image_file,
#                 "./../output_transform/" + image_file
#             )
#
#             transformation.transform_pipeline_rgb(
#                 "./../output_undistort/" + image_file,
#                 "./../output_transform_rgb/" + image_file
#             )
#     else:
#         image_files = os.listdir("./../output_lanes")
#         for image_file in image_files:
#             print("Untransforming " + image_file)
#             transformation.untransform_pipeline(
#                 "./../output_lanes/" + image_file,
#                 "./../output_undistort/" + image_file,
#                 "./../output_untransform/" + image_file
#             )