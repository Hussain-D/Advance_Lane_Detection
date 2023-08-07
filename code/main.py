"""
Main python module to execute the complete pipeline
"""
from camera_calibration import CameraCalibration
from image_transformation import ImageTransformation
from lane_detection import LaneDetector
import cv2
from moviepy.editor import VideoFileClip

if __name__ == "__main__":
    camera = CameraCalibration(9, 6)
    transform = ImageTransformation()
    lanes = LaneDetector()

    camera.calibrate()

    def process_image(image):
        """ Function to process image """
        # Undistort the image
        undist_image = camera.undistort_image(image)

        # Apply image transformations
        transformed_image = transform.binary_pipeline_image(
            undist_image, (40, 60), (220, 255), (90, 100)
        )
        perspective_binary = transform.transform_pipeline_image(transformed_image)

        # Detect lane lines and curvature
        lane_image, curvature, lane_center = lanes.pipeline_image(perspective_binary)

        # Untransform the image
        final_image = transform.untransform_pipeline_image(lane_image, image)

        # Add information on the images
        cv2.putText(
            final_image, f"Curvature: {round(curvature, 2)} m",
            (500, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 0), 2
        )
        
        actual_center = (image.shape[1] / 2) * (4.0 / image.shape[1])
        if lane_center < actual_center:
            cv2.putText(
                final_image, f"Offset: {round(actual_center - lane_center, 2)} m to right",
                (500, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2
            )
        else:
            cv2.putText(
                final_image, f"Offset: {round(lane_center - actual_center, 2)} m to left",
                (500, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2
            )

        return final_image

    
    output_video = "./../test_videos_output/project_video.mp4"
    clip1 = VideoFileClip("./../test_videos/challenge_video.mp4")
    project_clip = clip1.fl_image(process_image)
    project_clip.write_videofile(output_video, audio=False)
