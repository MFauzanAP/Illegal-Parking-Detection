import os
import numpy as np
import cv2 as cv

class MotionDetection():
	REPROJECT_THRESHOLD = 0.2
	DISPLAY_FLOW = "dense"
	FLOW_THRESHOLD = 17.5
	AREA_MIN_THRESHOLD = 5000			# Minimum area for a contour to be considered a car
	AREA_MAX_THRESHOLD = 25000			# Maximum area for a contour to be considered a car
	CIRCULARITY_MIN_THRESHOLD = 0.375	# Minimum circularity for a contour to be considered a car
	CIRCULARITY_MAX_THRESHOLD = 0.7		# Maximum circularity for a contour to be considered a car
	NUM_CLUSTERS = 6

	def __init__(self, dataset, img_width, output_flow=True, output_combined=True):
		self.dataset = dataset
		self.img_width = img_width
		self.output_flow = output_flow
		self.output_combined = output_combined

		self.detector = cv.SIFT_create()
		self.bf = cv.BFMatcher()

	# Analyze the image
	def analyze(self, img, img_index):
		self.img = img
		self.gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
		self.img_index = img_index

		# Find matches between two images and perform warping
		self.match_and_warp()

		# Calculate optical flow between the current and warped previous image
		if self.output_flow: self.plot_flow()

		# Combine the flow and warped along with matching features and save it
		if self.output_combined: self.plot_combined()

		# Set these variables for the next iteration
		self.old_img = self.img.copy()
		self.old_gray = self.gray.copy()
		self.kp1 = self.kp2
		self.des1 = self.des2

	# Shortcut for getting the path to a directory
	def get_path(self, dir, ext="jpg"): return f".\\output\\{self.dataset}\\{dir}\\{"{:05d}".format(self.img_index)}.{ext}"

	# Detect matches between two images and warp the previous image to the current image
	def match_and_warp(self):
		self.kp2, self.des2 = self.detector.detectAndCompute(self.gray, None)

		# Only proceed if we have keypoints in the previous image
		if self.img_index == 0 or len(self.kp1) == 0: return

		# Find matches between the two images and sort them by distance
		pair_matches = self.bf.knnMatch(self.des2, self.des1, k=2)
		self.matches = []
		for m, n in pair_matches:
			if m.distance < 0.7*n.distance:
				self.matches.append(m)
		self.matches = sorted(self.matches, key=lambda x: x.distance)

		# Only proceed if we have enough matches
		if len(self.matches) < 4: return print('Not enough matches found!')

		# Find homography matrix and do perspective transform
		src_pts = np.float32([self.kp1[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([self.kp2[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
		H, _ = cv.findHomography(src_pts, dst_pts, method=cv.FM_LMEDS, ransacReprojThreshold=self.REPROJECT_THRESHOLD, confidence=0.99)
		self.old_warped = cv.warpPerspective(self.old_gray, H, (self.gray.shape[1], self.gray.shape[0]))

		# If there are any black pixels in the warped image, replace them with the corresponding pixels in the new frame
		self.old_warped = np.where(self.old_warped <= 0, self.gray, self.old_warped)

	# Calculate optical flow between the current and warped previous image
	def plot_flow(self):
		# Only proceed if we have previous images
		if self.img_index == 0 or self.old_warped is None: return

		# Calculate dense optical flow
		flow = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(self.old_warped, self.gray, None)

		# Create image for displaying the flow
		mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
		hsv = np.zeros_like(self.img)
		hsv[..., 1] = 255
		hsv[..., 0] = ang*180/np.pi/2

		# Allow only flows that are above a certain threshold
		hsv[..., 2] = cv.threshold(mag, self.FLOW_THRESHOLD, 255, cv.THRESH_BINARY)[1]

		# Reduce the number of colors in the flow image to make it easier to identify distinct flows
		Z = hsv.reshape((-1, 3))
		Z = np.float32(Z)
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		_, label, center = cv.kmeans(Z, self.NUM_CLUSTERS, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
		center = np.uint8(center)
		res = center[label.flatten()]
		hsv = res.reshape((hsv.shape))

		# Extract the contours from the flow image to clean it up and draw bounding boxes around the blobs
		bboxes = []
		gray = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
		_, thresh = cv.threshold(gray, 175, 255, 0)
		contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		for contour in contours:

			# Calculate the area, perimeter, and circularity of the contour
			area = cv.contourArea(contour)
			perimeter = cv.arcLength(contour, True)
			circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

			# Filter out any blobs that are too small or too large, and those that are too thin or wide
			exceeds_area = area < self.AREA_MIN_THRESHOLD or area > self.AREA_MAX_THRESHOLD
			exceeds_circularity = circularity < self.CIRCULARITY_MIN_THRESHOLD or circularity > self.CIRCULARITY_MAX_THRESHOLD
			if exceeds_area or exceeds_circularity:
				cv.drawContours(hsv, [contour], -1, (0, 0, 0), -1)
			else:
				# Calculate the bounding box around the contour
				rect = cv.minAreaRect(contour)
				box = cv.boxPoints(rect)
				box = np.int0(box)
				bboxes.append(box)

		# Convert the flow to BGR for later use
		self.flow_img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
		cv.imwrite(self.get_path("flow-only"), self.flow_img)

		# Overlay the bounding boxes on the flow image
		img_bbox = self.img.copy()
		for bbox in bboxes:
			cv.drawContours(img_bbox, [bbox], 0, (255, 0, 0), 4)
		self.flow_bbox = img_bbox
		cv.imwrite(self.get_path("flow-bbox"), self.flow_bbox)

		# Overlay the flow on the original image
		self.combined_flow = cv.add(self.img, self.flow_img)
		cv.imwrite(self.get_path("flow"), self.combined_flow)

	# Combine the flow and warped along with matching features and save it
	def plot_combined(self):
		# Only proceed if we have previous images
		if self.img_index == 0 or self.old_warped is None: return

		# Draw the matching features on the image
		matching_features = cv.drawMatches(self.img, self.kp2, self.old_img, self.kp1, self.matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

		# Place images side by side
		output_img = np.vstack((
			np.hstack((self.old_img, cv.cvtColor(self.old_warped, cv.COLOR_GRAY2BGR), cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR))),
			np.hstack((matching_features, self.combined_flow))
		))

		# Save the combined image
		if self.output_combined: cv.imwrite(f'.\\output\\{self.dataset}\\combined\\{"{:05d}".format(self.img_index)}.jpg', output_img)
