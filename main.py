import os
import cv2 as cv
import argparse
import imutils
from defisheye import Defisheye

from lib.car_detection import CarDetection
from lib.geojson_plotter import GeoJsonPlotter
from lib.motion_detection import MotionDetection
from lib.ipc_detection import IPCDetection

# TODO:
# - Save screenshot, time, and approximate coordinates of the cars that are parked illegally as data in a point cloud
# - Find clusters of points in the point cloud that are close to each other and group them if they are the same car, do this by conducting image similarity analysis
# - For each group of data points, calculate the time the car has been parked there and the approximate coordinates of the car
# - If the cars have been parked there for more than a certain amount of time, flag them as illegally parked and save the data
# - Analyze images in parallel

RESIZED_WIDTH = 1000
OUTPUT_COMBINED = True
OUTPUT_FLOW = True
OUTPUT_CAMERA = False
OUTPUT_PARKING = True
OUTPUT_CARS = True
OUTPUT_DIRS = [
	'grid-resized',
	'cars',
	'cars-json',
	'cars-bbox',
	'cars-bbox-only',
	'combined-flow',
	'flow',
	'flow-only',
	'flow-bbox',
	'flow-bbox-only',
	'camera',
	'parking',
	'parking-bbox-only',
	'combined-ipc',
	'ipc',
	'ipc-only',
	'point-cloud'
]

# Has issues with plotting the camera, parking, and flow data, and also crops part of the image
REMOVE_DISTORTION = False
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
	car_detection = CarDetection(dataset=args.dataset, img_width=RESIZED_WIDTH, output_cars=OUTPUT_CARS)
	motion_detection = MotionDetection(dataset=args.dataset, img_width=RESIZED_WIDTH, output_flow=OUTPUT_FLOW, output_combined=OUTPUT_COMBINED)
	geojson_plotter = GeoJsonPlotter(dataset=args.dataset, img_width=RESIZED_WIDTH, output_camera=OUTPUT_CAMERA, output_parking=OUTPUT_PARKING)
	ipd_detection = IPCDetection(dataset=args.dataset, img_width=RESIZED_WIDTH, car_detection=car_detection, motion_detection=motion_detection, geojson_plotter=geojson_plotter)

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

		# Execute the pipeline components if they are enabled
		if OUTPUT_CARS: car_detection.analyze(img, img_index)
		if OUTPUT_FLOW or OUTPUT_COMBINED: motion_detection.analyze(img, img_index)
		if OUTPUT_CAMERA or OUTPUT_PARKING: geojson_plotter.analyze(img, img_index)
		if OUTPUT_CARS and OUTPUT_FLOW and OUTPUT_PARKING: ipd_detection.analyze(img, img_index)

		print(f'Frame {img_index + 1}/{num_images} processed ({round((img_index + 1) / num_images * 100, 2)}%)...')
		img_index += 1
	
	# Destroy the DarkHelp object
	car_detection.destroy()

if __name__ == "__main__":
    main()