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
3. Run the motion detection script: `python motion_detection.py "DATASET_PATH"`
4. View the results in the `output` directory

## Training the Model

1. Follow the steps given in the [Darknet Building Instructions](https://github.com/hank-ai/darknet/tree/master?tab=readme-ov-file#windows-cmake-method)
2. Fix CUDA compiler not found issue by following the steps here: [CUDA Compiler Not Found](https://stackoverflow.com/questions/56636714/cuda-compile-problems-on-windows-cmake-error-no-cuda-toolset-found)
3. Continue installation process
4. Build and install [DarkHelp](https://github.com/stephanecharette/DarkHelp?tab=readme-ov-file#building-darkhelp-windows)