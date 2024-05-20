import os
import cv2 as cv
import argparse
import imutils
from defisheye import Defisheye

from lib.car_detection import CarDetection
from lib.geojson_plotter import GeoJsonPlotter
from lib.motion_detection import MotionDetection
from lib.ipc_detection import IPCDetection
from lib.report_generator import ReportGenerator

# TODO:
# - Tweak flow detection to better detect cars even with small motion
# - Improve car detection model by training it on more data or data with only cars (to reduce false positives)
# - Analyze images in parallel
# - Create 3d model of each ipc if it has been parked there for a long time

RESIZED_WIDTH = 1000

# Output directories, comment out the ones that you don't need, check readme.md for more information
OUTPUT_DIRS = [
	'camera',
	'mission-path',
	'grid-resized',			# Required for car detection
	'cars',
	'cars-json',
	'cars-bbox',
	'cars-bbox-only',
	'flow',
	'flow-only',
	'flow-bbox',
	'flow-bbox-only',
	'parking',
	'parking-bbox-only',
	'ipc',
	'ipc-only',
	'tagged-ipc',
	'tagged-ipc-json',
	'point-cloud',
	'point-cloud-histogram',
	'clustered-point-cloud',
	'combined-flow',
	'combined-ipc',
	'combined-ipc-only',
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

	# Delete the old output directories
	if os.path.exists(f'.\\output\\{args.dataset}'):
		for dir in os.listdir(f'.\\output\\{args.dataset}'):
			if os.path.exists(f'.\\output\\{args.dataset}\\{dir}'):
				if os.path.isdir(f'.\\output\\{args.dataset}\\{dir}'):
					for file in os.listdir(f'.\\output\\{args.dataset}\\{dir}'):
						os.remove(f'.\\output\\{args.dataset}\\{dir}\\{file}')
					os.rmdir(f'.\\output\\{args.dataset}\\{dir}')
				else:
					os.remove(f'.\\output\\{args.dataset}\\{dir}')

	# Create output directories
	if not os.path.exists(f'.\\output\\{args.dataset}'): os.makedirs(f'.\\output\\{args.dataset}')
	for dir in OUTPUT_DIRS:
		if not os.path.exists(f'.\\output\\{args.dataset}\\{dir}'): os.makedirs(f'.\\output\\{args.dataset}\\{dir}')

	# Initialize the pipeline components
	car_detection = CarDetection(dataset=args.dataset, img_width=RESIZED_WIDTH)
	motion_detection = MotionDetection(dataset=args.dataset, img_width=RESIZED_WIDTH)
	geojson_plotter = GeoJsonPlotter(dataset=args.dataset, img_width=RESIZED_WIDTH)
	ipc_detection = IPCDetection(dataset=args.dataset, img_width=RESIZED_WIDTH, car_detection=car_detection, motion_detection=motion_detection, geojson_plotter=geojson_plotter)
	report_generator = ReportGenerator(dataset=args.dataset, img_width=RESIZED_WIDTH, car_detection=car_detection, motion_detection=motion_detection, geojson_plotter=geojson_plotter, ipc_detection=ipc_detection)

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
		car_detection.analyze(img, img_index)
		motion_detection.analyze(img, img_index)
		geojson_plotter.analyze(img, img_index)
		ipc_detection.analyze(img, img_index)

		print(f'Frame {img_index + 1}/{num_images} processed ({round((img_index + 1) / num_images * 100, 2)}%)...')
		img_index += 1

	print('All frames processed!')
	print('Analyzing point cloud data...')

	# Export camera data to a GeoJSON file
	geojson_plotter.export_camera_plot()

	# Export the point cloud data to a JSON file
	ipc_detection.export_point_cloud()

	# Cluster the point cloud data and analyze it to find the IPCs
	ipc_detection.analyze_point_cloud()

	# Generate the final report
	report_generator.export()

	# Destroy the DarkHelp object
	car_detection.destroy()

if __name__ == "__main__":
    main()