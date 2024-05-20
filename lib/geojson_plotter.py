import os
import re
import subprocess
import cv2 as cv
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cameratransform as ct

from eomaps import Maps
from datetime import datetime
from geojson import Polygon, LineString, dumps, loads

class GeoJsonPlotter():
	PARKING_TRANSPARENCY = 0.25
	METER_PER_RADIAN = 6371008.8

	def __init__(self, dataset, img_width):
		self.dataset = dataset
		self.img_width = img_width

		self.capture_points = []
		self.altitudes = []
		self.mission_start_time = None
		self.mission_end_time = None

		self.image_resolution = None
		self.focal_length = None
		self.mega_pixels = None
		self.capture_rate = 1
		self.field_of_view = None

		self.camera_plots = []
		self.min_lat = None
		self.max_lat = None
		self.min_lon = None
		self.max_lon = None

		# Load the parking geojson file
		self.load_parking()

	# Analyze the image
	def analyze(self, img, img_index):
		self.img = img
		self.img_index = img_index
		self.img_path = f"{os.getcwd()}\\data\\{self.dataset}\\{"{:05d}".format(img_index)}.jpg"
		self.parking_bboxes = []

		# Extract metadata from the image
		self.get_metadata()

		# Initialize the camera transform object
		self.init_camera()

		# Georeference the image to a geojson file
		self.plot_camera()

		# Plot the parking data on the image
		self.plot_parking()

	# Shortcut for getting the path to a directory
	def get_path(self, dir, ext="jpg"): return f".\\output\\{self.dataset}\\{dir}\\{"{:05d}".format(self.img_index)}.{ext}"

	# Exports all camera data to a final json file and plots it
	def export_camera_plot(self):

		# Write the camera polygons to a GeoJSON file
		try:
			with open(f".\\output\\{self.dataset}\\camera\\final.geojson", 'w') as f:
				f.write(dumps({
					'type': 'FeatureCollection',
					'features': [ { 'type': 'Feature', 'geometry': c, 'properties': {} } for c in self.camera_plots ]
				}))
		except: pass

		# Add some padding around map extent
		self.min_lat -= 0.00025
		self.max_lat += 0.00025
		self.min_lon -= 0.00025
		self.max_lon += 0.00025

		# Create map background
		m = Maps(crs=Maps.CRS.Mercator.GOOGLE, layer="base")
		m.new_layer("camera")
		m.new_layer("path")
		m.new_layer("satellite")
		m.set_extent((self.max_lon, self.min_lon, self.max_lat, self.min_lat))
		m.add_wms.ESRI_ArcGIS.SERVICES.World_Imagery.add_layer.xyz_layer(layer="satellite")

		# Plot this geojson polygon on a map
		df = gpd.read_file(f".\\output\\{self.dataset}\\camera\\final.geojson", driver='GeoJSON')
		m.add_gdf(df, layer="camera")
		m.show_layer(("satellite", 0.75), ("camera", 0.25))
		m.savefig(f".\\output\\{self.dataset}\\camera\\final.png")

		# Write the center points to a GeoJSON file
		line = LineString([ (p[1], p[0]) for p in self.capture_points ])
		try:
			with open(f".\\output\\{self.dataset}\\mission-path\\points.geojson", 'w') as f:
				f.write(dumps({
					'type': 'Feature',
					'geometry': line,
					'properties': {}
				}))
		except: pass

		# Plot this geojson polygon on a map
		df = gpd.read_file(f".\\output\\{self.dataset}\\mission-path\\points.geojson", driver='GeoJSON')
		m.add_gdf(df, layer="path")
		m.show_layer(("satellite", 0.75), "path")
		m.savefig(f".\\output\\{self.dataset}\\mission-path\\path.png")
		plt.clf()

	# Calculate the haversine distance between two points in meters
	def calculate_haversine(self, a, b):
		lat1, lon1 = np.radians(a)
		lat2, lon2 = np.radians(b)
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
		c = 2 * np.arcsin(np.sqrt(a))
		return self.METER_PER_RADIAN * c

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
			line = tag.strip().split(':', 1)
			info_dict[line[0].strip()] = line[-1].strip()

		# Get the timestamp data
		self.timestamp_data = int(datetime.strptime(info_dict['Date/Time Original'], "%Y:%m:%d %H:%M:%S").timestamp())

		# Get the camera data
		f = float(re.sub('[^0-9.-]', '', info_dict['Lens Info'][:6]))
		sx = 6.4		# https://sdk-forum.dji.net/hc/en-us/articles/12325496609689-What-is-the-custom-camera-parameters-for-Mavic-3-Enterprise-series-and-Mavic-3M
		sy = 4.8
		ix = self.img_width
		iy = int(info_dict['Image Height']) * ix // int(info_dict['Image Width'])
		self.camera_data = {'f': f, 'sx': sx, 'sy': sy, 'ix': ix, 'iy': iy}

		self.image_resolution = info_dict['Image Size']
		self.focal_length = f
		self.mega_pixels = info_dict['Megapixels']
		self.field_of_view = info_dict['Field Of View']

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

		# Update the mission start and end times
		if self.mission_start_time is None or self.timestamp_data < self.mission_start_time: self.mission_start_time = self.timestamp_data
		if self.mission_end_time is None or self.timestamp_data > self.mission_end_time: self.mission_end_time = self.timestamp_data

		# Update altitude list
		self.altitudes.append(alt)

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

		# Get the coordinates of the image corners and center
		center = self.cam.gpsFromImage([self.camera_data['ix']//2, self.camera_data['iy']//2])[:-1]
		coords = np.array([
			self.cam.gpsFromImage([0, 0]),
			self.cam.gpsFromImage([self.camera_data['ix']-1, 0]),
			self.cam.gpsFromImage([self.camera_data['ix']-1, self.camera_data['iy']-1]),
			self.cam.gpsFromImage([0, self.camera_data['iy']-1])
		])

		# Save the camera center to the list for calculating travel distance
		self.capture_points.append(center)
		if self.min_lat is None or center[0] < self.min_lat: self.min_lat = center[0]
		if self.max_lat is None or center[0] > self.max_lat: self.max_lat = center[0]
		if self.min_lon is None or center[1] < self.min_lon: self.min_lon = center[1]
		if self.max_lon is None or center[1] > self.max_lon: self.max_lon = center[1]

		# Create a GeoJSON polygon to represent the image corners
		points = []
		for point in coords:
			points.append((point[1], point[0]))
		points.append((coords[0][1], coords[0][0]))
		self.camera_polygon = Polygon([points], precision=8)

		# Write the camera polygon to a GeoJSON file
		self.camera_plots.append(self.camera_polygon)
		try:
			with open(self.get_path("camera", "geojson"), 'w') as f:
				f.write(dumps(self.camera_polygon))
		except: pass

	# Plot the parking data on the image
	def plot_parking(self):
		if self.parking is None: return

		# Project parking polygon coordinates to image coordinates
		self.parking_bbox_img = np.ones_like(self.img)
		self.parking_img = self.img.copy()
		for parking_lot in self.parking.get('features'):
			projected_coords = []
			for coord in parking_lot.get('geometry').get('coordinates')[0]:
				x, y = self.cam.imageFromGPS(np.array([coord[1], coord[0]]))
				projected_coords.append(np.int32([x, y]))
			self.parking_bboxes.append(np.array(projected_coords))

			# Draw the parking polygon on the image
			cv.fillPoly(self.parking_bbox_img, [np.array(projected_coords)], (0, 255, 0, self.PARKING_TRANSPARENCY * 255))

		# Overlay the parking mask on the image
		cv.imwrite(self.get_path("parking-bbox-only"), self.parking_bbox_img)
		cv.addWeighted(self.parking_bbox_img, self.PARKING_TRANSPARENCY, self.parking_img, 1, 0, self.parking_img)
		cv.imwrite(self.get_path("parking"), self.parking_img)
