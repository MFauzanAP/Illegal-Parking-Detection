import os
import json
import imutils
import cv2 as cv
import numpy as np

from lib import DarkHelp

class IPCDetection():

	def __init__(self, dataset, img_width, car_detection, motion_detection, geojson_plotter):
		self.dataset = dataset
		self.img_width = img_width
		self.car_detection = car_detection
		self.motion_detection = motion_detection
		self.geojson_plotter = geojson_plotter
		self.point_cloud = []

	# Call to destroy the DarkHelp object
	def destroy(self): DarkHelp.DestroyDarkHelpNN(self.dh)

	# Shortcut for getting the path to a directory
	def get_path(self, dir, ext="jpg", absolute=False): return f"{os.getcwd() if absolute == True else '.'}\\output\\{self.dataset}\\{dir}\\{"{:05d}".format(self.img_index)}.{ext}"

	# Analyze the image
	def analyze(self, img, img_index):
		self.img = img
		self.img_index = img_index

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

	# Export the point cloud data to a JSON file
	def export_point_cloud(self):
		try:
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
		for i, car in enumerate(self.cars_bboxes):
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
			car_img = self.img[y:y+h, x:x+w]
			if car_img.shape[0] > car_img.shape[1]: car_img = cv.transpose(car_img)
			self.ipc_imgs.append(car_img)

			# Save the car image to a file
			img_path = f".\\output\\{self.dataset}\\ipc-only\\{"{:05d}".format(self.img_index)}_{"{:05d}".format(num_illegal)}.jpg"
			cv.imwrite(img_path, car_img)

			# Store the time and geographic coordinates of the car
			gps_coords = self.cam.gpsFromImage([x + w/2, y + h/2])
			data = {
				"img": img_path,
				"time": self.timestamp,
				"lat": gps_coords[0],
				"lon": gps_coords[1],
				"maps": f"https://www.google.com/maps/search/{gps_coords[0]},{gps_coords[1]}/@{gps_coords[0]},{gps_coords[1]},20z?entry=ttu"
			}
			self.ipc_data.append(data)
			self.point_cloud.append(data)

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
