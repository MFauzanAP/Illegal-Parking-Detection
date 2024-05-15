# Illegal Parking Detection

## Introduction
This project aims to develop a system for detecting illegal parking using drones. The system will utilize computer vision techniques to identify vehicles parked in prohibited areas and send alerts to the authorities.

> **Note:** This project is developed as part of the course project for MECH420 - Introduction to Drones at Qatar University.

## Prerequisites
- Python 3.6 or higher
- [ExifTool](https://exiftool.org/) (extract to folder and add to PATH environment variable)

## Getting Started
1. Clone this repository: `git clone https://github.com/MFauzanAP/Illegal-Parking-Detection.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Open `main.py` and comment out any directories that are not needed in `OUTPUT_DIRS`. A description of each directory is given in [](#)
4. View the results in the `output` directory

## Training the Model

1. Follow the steps given in the [Darknet Building Instructions](https://github.com/hank-ai/darknet/tree/master?tab=readme-ov-file#windows-cmake-method)
2. Fix CUDA compiler not found issue by following the steps here: [CUDA Compiler Not Found](https://stackoverflow.com/questions/56636714/cuda-compile-problems-on-windows-cmake-error-no-cuda-toolset-found)
3. Continue installation process
4. Build and install [DarkHelp](https://github.com/stephanecharette/DarkHelp?tab=readme-ov-file#building-darkhelp-windows)

## Output Directories

The following list shows the available output directories and their descriptions. A sample output is provided in the `output` directory.

- `grid-resized`: This is required for the machine learning model to work. It contains the source images tiled into a 2x2 grid with black tiles.
- `cars`: Contains the output of the car detection model with annotated bounding boxes.
- `cars-json`: Output of the car detection model in JSON format.
- `cars-bbox`: Shows the source images with the bounding boxes drawn around the detected cars.
- `cars-bbox-only`: Black images with bounding boxes where cars were detected.
- `flow`: Processed optical flow images.
- `flow-bbox`: Source images with the bounding boxes drawn around areas of motion.
- `flow-bbox-only`: Black images with bounding boxes where motion was detected.
- `camera`: GeoJSON files with the camera locations and image corners.
- `parking`: Parking spots projected onto the source images.
- `parking-bbox-only`: Black images with bounding boxes where parking spots were detected.
- `ipc`: Source images with bounding boxes around the illegally parked cars drawn on them.
- `ipc-only`: Individual snippets of the illegally parked cars.
- `point-cloud`: Point cloud data for the illegally parked cars.
- `combined-flow`: Shows the warped images with feature tracks drawn and the final output flow image.
- `combined-ipc`: Overlays the car, flow, and parking bounding boxes into a single image.
- `combined-ipc-only`: Combines all IPC snippets in each frame into a single image
