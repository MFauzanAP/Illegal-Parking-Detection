import os
import numpy as np
import cv2 as cv
import argparse
import imutils
from defisheye import Defisheye

from lib.geojson_plotter import GeoJsonPlotter
from lib.motion_detection import MotionDetection

RESIZED_WIDTH = 1000
OUTPUT_DIRS = ['combined', 'flow', 'camera', 'parking']
OUTPUT_COMBINED = True
OUTPUT_FLOW = True
OUTPUT_CAMERA = True
OUTPUT_PARKING = True

REMOVE_DISTORTION = True
D_TYPE = 'stereographic'
D_FORMAT = 'fullframe'
D_FOV = 11
D_PFOV = 10

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str, help='Which dataset to load for analysis, this should be a subfolder name inside the "data" folder', default="output2")
	args = parser.parse_args()

	# Create output directories
	for dir in OUTPUT_DIRS:
		if not os.path.exists(f'.\\output\\{args.dataset}\\{dir}'): os.makedirs(f'.\\output\\{args.dataset}\\{dir}')

	# Initialize the pipeline components
	motion_detection = MotionDetection(dataset=args.dataset, img_width=RESIZED_WIDTH, output_flow=OUTPUT_FLOW, output_combined=OUTPUT_COMBINED)
	geojson_plotter = GeoJsonPlotter(dataset=args.dataset, img_width=RESIZED_WIDTH, output_camera=OUTPUT_CAMERA, output_parking=OUTPUT_PARKING)

	# Get the number of images in the directory
	cwd = os.getcwd()
	num_images = len([name for name in os.listdir(f"{cwd}\\data\\{args.dataset}") if os.path.isfile(os.path.join(f"{cwd}\\data\\{args.dataset}", name)) and name[-4:] == ".jpg"])

	# Keep looping through the images
	img_index = 0
	while(1):
		img_path = f"{cwd}\\data\\{args.dataset}\\{"{:05d}".format(img_index)}.jpg"

		# If the image does not exist, break the loop
		if not os.path.exists(img_path): break

		# Read the image and resize it
		img = cv.imread(img_path)
		img = imutils.resize(img, width=RESIZED_WIDTH)
		if REMOVE_DISTORTION: img = Defisheye(img, dtype=D_TYPE, format=D_FORMAT, fov=D_FOV, pfov=D_PFOV)._image

		# Execute the pipeline components
		motion_detection.analyze(img, img_index)
		geojson_plotter.analyze(img, img_index)

		print(f'Frame {img_index + 1}/{num_images} processed ({round((img_index + 1) / num_images * 100, 2)}%)...')
		img_index += 1

if __name__ == "__main__":
    main()