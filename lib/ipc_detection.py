import os
import json
import imutils
import contextlib
import cv2 as cv
import numpy as np

from lib import DarkHelp

class IPCDetection():
	

	def __init__(self, dataset, img_width):
		self.dataset = dataset
		self.img_width = img_width

	# Call to destroy the DarkHelp object
	def destroy(self): DarkHelp.DestroyDarkHelpNN(self.dh)

	# Shortcut for getting the path to a directory
	def get_path(self, dir, ext="jpg"): return f".\\output\\{self.dataset}\\{dir}\\{"{:05d}".format(self.img_index)}.{ext}"

	# Analyze the image
	def analyze(self, img, img_index, cars_bbox, flow_bbox, parking_bbox):
		self.img = img
		self.img_index = img_index
		self.cars_bbox = cars_bbox
		self.flow_bbox = flow_bbox
		self.parking_bbox = parking_bbox

		# combined = cv.add(self.cars_bbox, self.flow_bbox, self.parking_bbox)
		# cv.imshow('combined', combined)
		# cv.waitKey(0)

		# Filter out the cars that are moving or parked legally
		self.filter_cars()

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

	# Filter out the cars that are moving or parked legally
	def filter_cars(self):
		for car in self.cars_bbox:
			# Check if the car's bounding box intersects with any of the parking spots or flow contours
			is_legal = False
			is_moving = False
			for parking in self.parking_bbox:
				if self.detect_intersections(car, parking):
					print("Car is parked legally")
					is_legal = True
					break
			for flow in self.flow_bbox:
				if self.detect_intersections(car, flow):
					print("Car is moving")
					is_moving = True
					break
			
			# Take snapshot of the car if it is not moving and parked illegally
			if not is_legal and not is_moving:
				x, y, w, h = cv.boundingRect(car)
				car_img = self.img[y:y+h, x:x+w]
				cv.imshow('car', car_img)
				cv.waitKey(0)
