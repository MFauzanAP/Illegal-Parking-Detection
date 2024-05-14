import os
import subprocess
import numpy as np
import cv2 as cv
import argparse
import imutils

from PIL import Image
from PIL.ExifTags import TAGS

parser = argparse.ArgumentParser(
	description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
	The example file can be downloaded from: \
	https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4'
)
parser.add_argument('images', type=str, help='path to image files', default="\\data")
args = parser.parse_args()

# Create output directory
if not os.path.exists(f'.\\output\\{args.images}'): os.makedirs(f'.\\output\\{args.images}')
if not os.path.exists(f'.\\combined-output\\{args.images}'): os.makedirs(f'.\\combined-output\\{args.images}')

cap = cv.VideoCapture(f".\\data\\{args.images}\\%5d.jpg")
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1

cwd = os.getcwd()

img_index = 0
while(1):

	# Create process to use ExifTool
	print(f"{cwd}\\data\\{args.images}\\00000.jpg")
	process = subprocess.Popen(['exiftool', f"{cwd}\\data\\{args.images}\\00000.jpg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

	# Loop through each tag in the exif data
	info_dict = {}
	for tag in process.stdout:
		line = tag.strip().split(':')
		info_dict[line[0].strip()] = line[-1].strip()
	# for k,v in info_dict.items(): print(k,':', v)

	# # Open the image file
	# image = Image.open(f".\\data\\{args.images}\\{img_index:05d}.jpg")

	# # Extract the exif data
	# exif_data = image._getexif()

	# # Iterate over the exif data and print the tags
	# for id in exif_data:
	# 	tag_name = TAGS.get(id, id)
	# 	value = exif_data.get(id)
	# 	print(f"{tag_name:25}: {value}")

	img_index += 1
	break
