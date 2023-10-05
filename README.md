# Advance Lane Detection

[![GitHub stars](https://img.shields.io/github/stars/Hussain-D/Advance_Lane_Detection?style=social)](https://github.com/Hussain-D/Advance_Lane_Detection/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Hussain-D/Advance_Lane_Detection)](https://github.com/Hussain-D/Advance_Lane_Detection/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Hussain-D/Advance_Lane_Detection/blob/main/LICENSE)

## Overview

The **Advance Lane Detection** project is an advanced computer vision system designed to accurately detect and visualize lane lines on the road. It's a critical component of autonomous vehicles and advanced driver-assistance systems (ADAS). This project implements a robust lane detection pipeline that works in various driving conditions, including challenging scenarios such as curved roads and varying lighting conditions.

## Features

- **Lane Detection:** Accurately detects and highlights lane lines on the road.
- **Lane Curvature:** Calculates and displays lane curvature information.
- **Lane Centering:** Determines the position of the vehicle relative to the center of the lane.
- **Visualization:** Visualizes the detected lane lines and curvature on the video frames.
- **Advanced Lane Finding:** Implements advanced techniques such as perspective transform and color thresholding for robust lane detection.

## Dependencies

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
```bash
  git clone https://github.com/Hussain-D/Advance_Lane_Detection.git
  cd Advance_Lane_Detection
```
2. Install the required dependencies:
You can install these dependencies using the provided `requirements.txt` file:
```bash
  pip install -r requirements.txt
```
3. Usage
To use the lane detection system, follow these steps:

Run the lane detection on a video:
```bash
  python lane_detection.py --input video.mp4 --output output.mp4
```
Replace video.mp4 with the path to your input video file and output.mp4 with the desired output video filename.
The processed video with lane detection results will be saved in the specified output file.

[Demo Output Video](test_videos_output/project_video.mp4)

Click the link above to watch a demo video showcasing the lane detection system in action.

## Contributing

Contributions are welcome! Feel free to open issues and pull requests for any improvements or bug fixes.
License

This project is licensed under the [MIT License](https://github.com/Hussain-D/Advance_Lane_Detection/blob/main/LICENSE) - see the [LICENSE](LICENSE) file for details.

This enhanced README provides a comprehensive overview of your Blogging Website project, including its features, installation instructions, screenshots, and information on how others can contribute. Make sure to replace the placeholder links and filenames with the actual ones from your project.

