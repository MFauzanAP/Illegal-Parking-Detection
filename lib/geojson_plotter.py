import os
import re
import subprocess
import cv2 as cv
import numpy as np
import cameratransform as ct

from geojson import Polygon, dumps, loads

class GeoJsonPlotter():
	PARKING_TRANSPARENCY = 0.25

	def __init__(self, dataset, img_width, output_camera=True, output_parking=True):
		self.dataset = dataset
		self.img_width = img_width
		self.output_camera = output_camera
		self.output_parking = output_parking

		# Load the parking geojson file
		self.load_parking()

	# Analyze the image
	def analyze(self, img, img_index):
		self.img = img
		self.img_index = img_index
		self.img_path = f"{os.getcwd()}\\data\\{self.dataset}\\{"{:05d}".format(img_index)}.jpg"

		# Extract metadata from the image
		self.get_metadata()

		# Initialize the camera transform object
		self.init_camera()

		# Georeference the image to a geojson file
		if self.output_camera: self.plot_camera()

		# Plot the parking data on the image
		if self.output_parking: self.plot_parking()

	# Shortcut for getting the path to a directory
	def get_path(self, dir, ext="jpg"): return f".\\output\\{self.dataset}\\{dir}\\{"{:05d}".format(self.img_index)}.{ext}"

	# Load the parking geojson file
	def load_parking(self):
		try:
			with open(f'.\\data\\{self.dataset}\\parking.geojson', 'r') as f:
				self.parking = loads(f.read())
		except:
			print("No parking data found for this dataset")
		
	# Extract metadata from the image
	def get_metadata(self, verbose=False):
		# Create process to use ExifTool
		process = subprocess.Popen(['exiftool', self.img_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

		# Loop through each tag in the exif data
		info_dict = {}
		for tag in process.stdout:
			line = tag.strip().split(':')
			info_dict[line[0].strip()] = line[-1].strip()

		# Get the camera data
		f = float(re.sub('[^0-9.-]', '', info_dict['Lens Info'][:6]))
		sx = 6.4		# https://sdk-forum.dji.net/hc/en-us/articles/12325496609689-What-is-the-custom-camera-parameters-for-Mavic-3-Enterprise-series-and-Mavic-3M
		sy = 4.8
		ix = self.img_width
		iy = int(info_dict['Image Height']) * ix // int(info_dict['Image Width'])
		self.camera_data = {'f': f, 'sx': sx, 'sy': sy, 'ix': ix, 'iy': iy}

		# Get the GPS data
		gps = info_dict['GPS Position'].replace(' deg', '°')
		lat, lon = ct.gpsFromString(gps)
		alt = float(re.sub('[^0-9.-]', '', info_dict['Relative Altitude']))
		self.gps_data = {'lat': lat, 'lon': lon, 'alt': alt}

		# Get the orientation data
		drone_roll = float(re.sub('[^0-9.-]', '', info_dict['Flight Roll Degree']))
		drone_yaw = float(re.sub('[^0-9.-]', '', info_dict['Flight Yaw Degree']))
		drone_pitch = 90 + float(re.sub('[^0-9.-]', '', info_dict['Flight Pitch Degree']))
		gimbal_roll = float(re.sub('[^0-9.-]', '', info_dict['Gimbal Roll Degree']))
		gimbal_yaw = float(re.sub('[^0-9.-]', '', info_dict['Gimbal Yaw Degree']))
		gimbal_pitch = 90 + float(re.sub('[^0-9.-]', '', info_dict['Gimbal Pitch Degree']))
		self.orientation_data = {
			'drone_roll': drone_roll,
			'drone_yaw': drone_yaw,
			'drone_pitch': drone_pitch,
			'gimbal_roll': gimbal_roll,
			'gimbal_yaw': gimbal_yaw,
			'gimbal_pitch': gimbal_pitch
		}

		if verbose:
			for k,v in info_dict.items(): print(k,':', v)
			print("Focal Length:", f, "Sensor Width:", sx, "Sensor Height:", sy, "Image Width:", ix, "Image Height:", iy)
			print("Latitude:", lat, "Longitude:", lon, "Altitude:", alt)
			print("Drone Roll:", drone_roll, "Drone Yaw:", drone_yaw, "Drone Pitch:", drone_pitch)
			print("Gimbal Roll:", gimbal_roll, "Gimbal Yaw:", gimbal_yaw, "Gimbal Pitch:", gimbal_pitch)

	# Initialize the camera transform object
	def init_camera(self):
		# Unpack the camera, GPS, and orientation data
		f = self.camera_data['f']
		sx = self.camera_data['sx']
		sy = self.camera_data['sy']
		ix = self.camera_data['ix']
		iy = self.camera_data['iy']
		lat = self.gps_data['lat']
		lon = self.gps_data['lon']
		alt = self.gps_data['alt']
		gimbal_roll = self.orientation_data['gimbal_roll']
		gimbal_yaw = self.orientation_data['gimbal_yaw']
		gimbal_pitch = self.orientation_data['gimbal_pitch']

		self.cam = ct.Camera(
			ct.RectilinearProjection(focallength_mm=f, sensor=(sx, sy), image=(ix, iy)),
			ct.SpatialOrientation(elevation_m=alt, tilt_deg=gimbal_pitch, roll_deg=gimbal_roll, heading_deg=gimbal_yaw)
		)
		self.cam.setGPSpos(lat, lon)

	# Georeference the image to a geojson file
	def plot_camera(self):
		# Get the coordinates of the image corners
		coords = np.array([
			self.cam.gpsFromImage([0, 0]),
			self.cam.gpsFromImage([self.camera_data['ix']-1, 0]),
			self.cam.gpsFromImage([self.camera_data['ix']-1, self.camera_data['iy']-1]),
			self.cam.gpsFromImage([0, self.camera_data['iy']-1])
		])

		# Create a GeoJSON polygon to represent the image corners
		points = []
		for point in coords:
			points.append((point[1], point[0]))
		points.append((coords[0][1], coords[0][0]))
		self.camera_polygon = Polygon([points], precision=8)

		# Write the camera polygon to a GeoJSON file
		with open(self.get_path("camera", "geojson"), 'w') as f:
			f.write(dumps(self.camera_polygon))

	# Plot the parking data on the image
	def plot_parking(self):
		# Project parking polygon coordinates to image coordinates
		projected_coords = []
		for coord in self.parking.get('features')[0].get('geometry').get('coordinates')[0]:
			x, y = self.cam.imageFromGPS(np.array([coord[1], coord[0]]))
			projected_coords.append([x, y])

		# Draw the parking polygon on the image and save it
		self.parking_mask = np.ones_like(self.img)
		cv.fillPoly(self.parking_mask, [np.array(projected_coords, np.int32)], (0, 255, 0, self.PARKING_TRANSPARENCY * 255))
		cv.imwrite(self.get_path("parking-only"), self.parking_mask)

		# Overlay the parking mask on the image
		cv.addWeighted(self.parking_mask, self.PARKING_TRANSPARENCY, self.img, 1, 0, self.img)
		cv.imwrite(self.get_path("parking"), self.img)
