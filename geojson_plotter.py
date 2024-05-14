import os
import re
import subprocess
import numpy as np
import cv2 as cv
import argparse
import cameratransform as ct

from lat_lon_parser import parse
from geojson import Polygon, dumps

parser = argparse.ArgumentParser(
	description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
	The example file can be downloaded from: \
	https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4'
)
parser.add_argument('images', type=str, help='path to image files', default="\\data")
args = parser.parse_args()

# Create output directory
if not os.path.exists(f'.\\output\\geojson\\{args.images}'): os.makedirs(f'.\\output\\geojson\\{args.images}')

# Get the number of images in the directory
cwd = os.getcwd()
num_images = len([name for name in os.listdir(f"{cwd}\\data\\{args.images}") if os.path.isfile(os.path.join(f"{cwd}\\data\\{args.images}", name))])
# process = subprocess.Popen(['exiftool', f"{cwd}\\data\\{args.images}\\00000.jpg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

img_index = 0
while(1):

	# If the image does not exist, break the loop
	if not os.path.exists(f"{cwd}\\data\\{args.images}\\{"{:05d}".format(img_index)}.jpg"): break

	# Create process to use ExifTool
	process = subprocess.Popen(['exiftool', f"{cwd}\\data\\{args.images}\\{"{:05d}".format(img_index)}.jpg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

	# Loop through each tag in the exif data
	info_dict = {}
	for tag in process.stdout:
		line = tag.strip().split(':')
		info_dict[line[0].strip()] = line[-1].strip()
	# for k,v in info_dict.items(): print(k,':', v)

	# Get the camera data
	f = float(re.sub('[^0-9.-]', '', info_dict['Focal Length']))
	sx = 6.4		# https://sdk-forum.dji.net/hc/en-us/articles/12325496609689-What-is-the-custom-camera-parameters-for-Mavic-3-Enterprise-series-and-Mavic-3M
	sy = 4.8
	ix = int(info_dict['Image Width'])
	iy = int(info_dict['Image Height'])

	# Get the GPS data
	lat = parse(info_dict['GPS Latitude'])
	lon = parse(info_dict['GPS Longitude'])
	alt = float(re.sub('[^0-9.-]', '', info_dict['GPS Altitude']))
	# print("Latitude:", lat, "Longitude:", lon, "Altitude:", alt)

	# Get the orientation data
	drone_roll = float(re.sub('[^0-9.-]', '', info_dict['Flight Roll Degree']))
	drone_yaw = float(re.sub('[^0-9.-]', '', info_dict['Flight Yaw Degree']))
	drone_pitch = 90 + float(re.sub('[^0-9.-]', '', info_dict['Flight Pitch Degree']))
	gimbal_roll = float(re.sub('[^0-9.-]', '', info_dict['Gimbal Roll Degree']))
	gimbal_yaw = float(re.sub('[^0-9.-]', '', info_dict['Gimbal Yaw Degree']))
	gimbal_pitch = 90 + float(re.sub('[^0-9.-]', '', info_dict['Gimbal Pitch Degree']))
	# print("Drone Roll:", drone_roll, "Drone Yaw:", drone_yaw, "Drone Pitch:", drone_pitch)
	# print("Gimbal Roll:", gimbal_roll, "Gimbal Yaw:", gimbal_yaw, "Gimbal Pitch:", gimbal_pitch)

	# Initialize the camera transform object
	cam = ct.Camera(
		ct.RectilinearProjection(focallength_mm=f, sensor=(sx, sy), image=(ix, iy)),
		ct.SpatialOrientation(elevation_m=alt, tilt_deg=gimbal_pitch, roll_deg=drone_roll+gimbal_roll, heading_deg=gimbal_yaw)
	)
	cam.setGPSpos(lat, lon, alt)

	# Get the coordinates of the image corners
	coords = np.array([cam.gpsFromImage([0, 0]), cam.gpsFromImage([ix-1, 0]), cam.gpsFromImage([ix-1, iy-1]), cam.gpsFromImage([0, iy-1])])
	# coords = cam.imageFromGPS(np.array([25.384456, 51.489702, 0]))
	# print(coords)

	# Create a GeoJSON polygon to represent the image corners
	points = []
	for point in coords:
		points.append((point[1], point[0]))
	points.append((coords[0][1], coords[0][0]))
	polygon = Polygon([points], precision=8)

	# Write the GeoJSON polygon to a file
	with open(f'.\\output\\geojson\\{args.images}\\{"{:05d}".format(img_index)}.geojson', 'w') as f:
		f.write(dumps(polygon))

	print(f'Frame {img_index + 1}/{num_images} processed ({round((img_index + 1) / num_images * 100, 2)}%)...')
	img_index += 1
	# break
