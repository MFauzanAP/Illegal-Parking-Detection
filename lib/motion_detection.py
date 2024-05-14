import os
import numpy as np
import cv2 as cv
import argparse
import imutils

from defisheye import Defisheye

class MotionDetection():
	REPROJECT_THRESHOLD = 0.2
	DISPLAY_FLOW = "dense"
	FLOW_NORMALIZATION = "none" # "none", "min", "max"
	FLOW_THRESHOLD = 50

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
		self.img_path = f"{os.getcwd()}\\data\\{self.dataset}\\{"{:05d}".format(img_index)}.jpg"

		# Find matches between two images and perform warping
		self.match_and_warp()

		# Calculate optical flow between the current and warped previous image
		self.plot_flow()

		# Combine the flow and warped along with matching features and save it
		self.plot_combined()

		# Set these variables for the next iteration
		self.old_img = self.img.copy()
		self.old_gray = self.gray.copy()
		self.kp1 = self.kp2
		self.des1 = self.des2

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
		for i in range(self.old_warped.shape[0]):
			for j in range(self.old_warped.shape[1]):
				if self.old_warped[i, j] <= 0:
					self.old_warped[i, j] = self.gray[i, j]

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

		# Normalize the flow
		if self.FLOW_NORMALIZATION != "none":
			for i in range(mag.shape[0]):
				for j in range(mag.shape[1]):
					if self.FLOW_NORMALIZATION == "min" and mag[i, j] < self.FLOW_THRESHOLD:
						mag[i, j] = 0
					elif self.FLOW_NORMALIZATION == "max" and mag[i, j] > self.FLOW_THRESHOLD:
						mag[i, j] = 0
			hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
		else: hsv[..., 2] = mag

		# Convert the flow to BGR for later use
		self.flow_img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
		self.combined_flow = cv.add(self.img, self.flow_img)

		# Save the flow image
		if self.output_flow: cv.imwrite(f'.\\output\\{self.dataset}\\flow\\{"{:05d}".format(self.img_index)}.jpg', self.combined_flow)

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
