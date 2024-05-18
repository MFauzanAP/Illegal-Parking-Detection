import os
import json
import imutils
import cv2 as cv
import numpy as np

from lib import DarkHelp
from perception import hashers
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

class IPCDetection():
	MIN_DISTANCE = 1					# in meters
	MIN_SIMILARITY = 120
	MIN_SAMPLES = 10
	METER_PER_RADIAN = 6371008.8

	MAX_ILLEGAL_PARKING_TIME = 15		# in seconds, how long can a car be parked illegally before it is tagged for investigation

	def __init__(self, dataset, img_width, car_detection, motion_detection, geojson_plotter):
		self.dataset = dataset
		self.img_width = img_width
		self.car_detection = car_detection
		self.motion_detection = motion_detection
		self.geojson_plotter = geojson_plotter

		self.detector = motion_detection.detector
		self.bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

		self.point_cloud = []
		self.point_cloud_imgs = []

	# Call to destroy the DarkHelp object
	def destroy(self): DarkHelp.DestroyDarkHelpNN(self.dh)

	# Shortcut for getting the path to a directory
	def get_path(self, dir, ext="jpg", absolute=False): return f"{os.getcwd() if absolute == True else '.'}\\output\\{self.dataset}\\{dir}\\{"{:05d}".format(self.img_index)}.{ext}"

	# Analyze the image
	def analyze(self, img, img_index):
		self.img = img
		self.img_index = img_index
		self.original_img = cv.imread(f".\\data\\{self.dataset}\\{"{:05d}".format(img_index)}.jpg")
		self.img_scale = self.original_img.shape[1] // self.img.shape[1]

		# Only proceed if we have previous images
		if self.img_index == 0: return

		# Get the data from the pipeline components
		self.car_bbox_img = self.car_detection.car_bbox_img
		self.flow_bbox_img = self.motion_detection.flow_bbox_img
		self.parking_bbox_img = self.geojson_plotter.parking_bbox_img
		self.cars_bboxes = self.car_detection.car_bboxes
		self.flow_bboxes = self.motion_detection.flow_bboxes
		self.parking_bboxes = self.geojson_plotter.parking_bboxes
		self.cam = self.geojson_plotter.cam
		self.timestamp = self.geojson_plotter.timestamp_data

		# Combine and overlay the camera, parking, and flow data on the image
		self.plot_combined()

		# Plot the point cloud of the cars that are parked illegally
		self.plot_point_cloud()

	# Export the point cloud data to a JSON and GeoJSON file
	def export_point_cloud(self):
		try:
			geojson = {
				"type": "FeatureCollection",
				"features": []
			}
			for data in self.point_cloud:
				geojson["features"].append({
					"type": "Feature",
					"geometry": {
						"type": "Point",
						"coordinates": [data["lon"], data["lat"]]
					},
					"properties": {
						"img": data["img"],
						"time": data["time"],
						"maps": data["maps"]
					}
				})
			with open(f".\\output\\{self.dataset}\\point-cloud\\final.geojson", 'w') as f:
				f.write(json.dumps(geojson, indent=4))
			with open(f".\\output\\{self.dataset}\\point-cloud\\final.json", 'w') as f:
				f.write(json.dumps(self.point_cloud, indent=4))
		except: pass

	# Detects intersections between two contours, or if they are overlapping
	def detect_intersections(self, c1, c2):
		def ccw(A, B, C): return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
		for i in range(len(c1)-1):
			a1 = c1[i]
			b1 = c1[i + 1]
			for j in range(len(c2)-1):
				a2 = c2[j]
				b2 = c2[j + 1]

				# Check if any line in both contours intersect with each other
				if ccw(a1,a2,b2) != ccw(b1,a2,b2) and ccw(a1,b1,a2) != ccw(a1,b1,b2): return True

				# Check if a point in one contour is inside the other contour
				if cv.pointPolygonTest(c1, np.float16(a2), False) >= 0 or cv.pointPolygonTest(c2, np.float16(a1), False) >= 0: return True

		# If no intersections are found, return False
		return False

	# Combine and overlay the camera, parking, and flow data on the image
	def plot_combined(self):
		self.combined = cv.addWeighted(self.img, 1, self.car_bbox_img, 0.25, 0)
		self.combined = cv.addWeighted(self.combined, 1, self.flow_bbox_img, 0.25, 0)
		self.combined = cv.addWeighted(self.combined, 1, self.parking_bbox_img, 0.25, 0)
		cv.imwrite(self.get_path("combined-ipc"), self.combined)

	# Filter out the cars that are moving or parked legally and plot the rest in the point cloud, with screenshot, time, and approximate coordinates
	def plot_point_cloud(self):
		self.ipc_imgs = []
		self.ipc_data = []
		self.ipc_bboxes = []
		num_illegal = 0
		for car in self.cars_bboxes:
			# Check if the car's bounding box intersects with any of the parking spots or flow contours
			is_legal = False
			is_moving = False
			for parking in self.parking_bboxes:
				if self.detect_intersections(car, parking):
					is_legal = True
					break
			for flow in self.flow_bboxes:
				if self.detect_intersections(car, flow):
					is_moving = True
					break

			# If the car is moving or parked legally, skip it
			if is_legal or is_moving: continue

			# Add the car's bounding box to the list
			self.ipc_bboxes.append(car)

			# Take snapshot of the car if it is not moving and parked illegally and add it to the list, and rotate it if necessary
			x, y, w, h = cv.boundingRect(car)
			x_index = np.int0(x * self.img_scale)
			y_index = np.int0(y * self.img_scale)
			w_index = np.int0(w * self.img_scale)
			h_index = np.int0(h * self.img_scale)
			car_img = self.original_img[y_index:y_index+h_index, x_index:x_index+w_index]
			if car_img.shape[0] > car_img.shape[1]: car_img = cv.transpose(car_img)
			self.ipc_imgs.append(car_img)

			# Save the car image to a file
			img_path = f".\\output\\{self.dataset}\\ipc-only\\{"{:05d}".format(self.img_index)}_{"{:05d}".format(num_illegal)}.jpg"
			cv.imwrite(img_path, car_img)

			# TODO: if more points are needed, either use the coords of each corner of the bounding box, or
			# extract good features from the image and use those as points, this may be needed to construct 3d model

			# Store the time and geographic coordinates of the car
			gps_coords = self.cam.gpsFromImage([x + w/2, y + h/2])
			data = {
				"img": img_path,
				"time": int(self.timestamp),
				"lat": gps_coords[0],
				"lon": gps_coords[1],
				"maps": f"https://www.google.com/maps/search/{gps_coords[0]},{gps_coords[1]}/@{gps_coords[0]},{gps_coords[1]},20z?entry=ttu"
			}
			self.ipc_data.append(data)
			self.point_cloud.append(data)
			self.point_cloud_imgs.append(car_img)

			# Increment the number of illegal cars
			num_illegal += 1

		# If there are no cars parked illegally, return
		if num_illegal == 0: return

		# Draw the bounding boxes of the illegal cars on the image
		output_ipc_img = self.img.copy()
		for bbox in self.ipc_bboxes:
			cv.drawContours(output_ipc_img, [bbox], 0, (0, 0, 255), 4)
		cv.imwrite(self.get_path("ipc"), output_ipc_img)

		# Combine the car images into a single image
		max_w = max([car_img.shape[1] for car_img in self.ipc_imgs])
		combined = cv.vconcat([imutils.resize(car_img, width=max_w) for car_img in self.ipc_imgs])
		cv.imwrite(self.get_path("combined-ipc-only"), combined)

		# Save the point cloud data to a JSON file
		try:
			with open(self.get_path("point-cloud", "json"), 'w') as f:
				f.write(json.dumps(self.ipc_data, indent=4))
		except: pass

	# Calculate the haversine distance between two points in kilometers
	def calculate_haversine(self, a, b):
		lat1, lon1, _, _ = a
		lat2, lon2, _, _ = b
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
		c = 2 * np.arcsin(np.sqrt(a))
		return self.METER_PER_RADIAN * c
	
	# Calculate similarity between two images based on feature matching, color histograms and perceptual hash
	def calculate_similarity(self, a, b):
		_, _, _, i = a
		_, _, _, j = b
		img1 = np.ascontiguousarray(self.point_cloud_imgs[int(i)])
		img2 = np.ascontiguousarray(self.point_cloud_imgs[int(j)])
		img2 = imutils.resize(img2, height=img1.shape[0], width=img1.shape[1])

		# Perform feature matching and get the number of matches
		_, des1 = self.detector.detectAndCompute(img1, None)
		_, des2 = self.detector.detectAndCompute(img2, None)
		matches = sorted(self.bf.match(des2, des1), key=lambda x: x.distance)
		num_matches = len(matches)

		# Calculate color histograms and get the correlation
		hist1 = cv.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist2 = cv.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist1 = cv.normalize(hist1, hist1).flatten()
		hist2 = cv.normalize(hist2, hist2).flatten()
		correlation = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)

		# Calculate the perceptual hash and get the hamming distance
		hasher = hashers.PHash()
		hash1 = hasher.compute(img1)
		hash2 = hasher.compute(img2)
		hd = hasher.compute_distance(hash1, hash2)
		if hd == 0: hd = 0.0001

		# Return the similarity score
		return num_matches * correlation / hd

	# Find clusters of points in the point cloud based on image similarity and distance
	def cluster_point_cloud(self):
		if len(self.point_cloud) == 0: return

		# Define the metric for the DBSCAN algorithm
		def metric(a, b):
			# Calculate the distance between two points, and if it is greater than the minimum distance, skip the rest of the calculation
			haversine = self.calculate_haversine(a, b)
			if haversine > self.MIN_DISTANCE: return haversine * self.MIN_SIMILARITY

			# Calculate the similarity between two images
			similarity = self.calculate_similarity(a, b)
			return haversine / similarity

		# Create a matrix of coordinates for each car
		points = np.array([[car["lat"], car["lon"], car["time"], i] for i, car in enumerate(self.point_cloud)])

		# Perform DBSCAN clustering on the point cloud
		db = DBSCAN(
			eps=self.MIN_DISTANCE / self.MIN_SIMILARITY,
			min_samples=self.MIN_SAMPLES,
			metric=metric
		)
		db.fit(np.concatenate([np.radians([x for x in zip(points[:, 0], points[:, 1])]), points[:, 2:].reshape(-1, 2)], axis=1))
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)

		# Extract the clusters and create a mask
		unique_labels = set(labels)
		core_samples_mask = np.zeros_like(labels, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True

		# Plot the clustered point cloud whilst saving them to a list
		ipc_cluster_points = []
		colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
		for k, col in zip(unique_labels, colors):
			if k == -1: col = [0, 0, 0, 1]
			class_member_mask = labels == k

			# Get all the points in the cluster and add it to the list
			cluster_points = points[class_member_mask]
			if len(cluster_points) > 0 and k != -1:
				original_points = [self.point_cloud[int(i)] for i in cluster_points[:, 3]]
				ipc_cluster_points.append(original_points)

			# Core points
			core_points = points[class_member_mask & core_samples_mask]
			plt.plot(
				core_points[:, 0],
				core_points[:, 1],
				"o",
				markerfacecolor=tuple(col),
				markeredgecolor="k",
				markersize=14,
			)

			# None-core / noise points
			noise_points = points[class_member_mask & ~core_samples_mask]
			plt.plot(
				noise_points[:, 0],
				noise_points[:, 1],
				"o",
				markerfacecolor=tuple(col),
				markeredgecolor="k",
				markersize=6,
			)

		# Save the clustered point cloud to a file
		plt.title(f"Estimated Number of IPCs: {n_clusters_}, Estimated number of noise points: {n_noise_}")
		plt.savefig(self.get_path("clustered-point-cloud", "png"))
		# plt.show()
		
		# Calculate the amount of time each car has been parked illegally in seconds and only tag the cars that have been parked illegally for too long
		for ipc_points in ipc_cluster_points:
			min_time = min([int(ipc["time"]) for ipc in ipc_points])
			max_time = max([int(ipc["time"]) for ipc in ipc_points])
			parking_time = max_time - min_time

			if parking_time < self.MAX_ILLEGAL_PARKING_TIME: continue

			# Display the image of the car that has been parked illegally for too long
			cv.imshow(f"Parking Time: {parking_time} seconds", cv.imread(ipc_points[0]["img"]))
			cv.waitKey(0)

	# Analyze each cluster and calculate how much time each car has been parked illegally
	def analyze_clusters(self):
		pass
