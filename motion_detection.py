import os
import numpy as np
import cv2 as cv
import argparse
import imutils
from defisheye import Defisheye

RESIZED_WIDTH = 1000
RESIZED_WIDTH_VIEWING = 300
REPROJECT_THRESHOLD = 0.2
DISPLAY_FLOW = "dense"

dtype = 'stereographic'
format = 'fullframe'
fov = 11
pfov = 10

parser = argparse.ArgumentParser(
    description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
	The example file can be downloaded from: \
	https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4'
)
parser.add_argument('images', type=str, help='path to image files', default="\\data")
args = parser.parse_args()

# Create output directory
if not os.path.exists(f'.\\output\\{args.images}'):
	os.makedirs(f'.\\output\\{args.images}')

cap = cv.VideoCapture(f".\\data\\{args.images}\\%5d.jpg")
detector = cv.SIFT_create(700)
bf = cv.BFMatcher()

# Params for removing distortion
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# params for ShiTomasi corner detection
feature_params = dict(
    maxCorners = 100,
	qualityLevel = 0.3,
	minDistance = 7,
	blockSize = 7
)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = Defisheye(old_frame, dtype=dtype, format=format, fov=fov, pfov=pfov)._image
old_frame = imutils.resize(old_frame, width=RESIZED_WIDTH)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
kp1, des1 = detector.detectAndCompute(old_gray, None)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)
mask_sparse = np.zeros_like(old_frame)

hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255

img_index = 0
while(1):
	img_index += 1
	ret, frame = cap.read()
	if not ret:
		print('No frames grabbed!')
		break

	frame = Defisheye(frame, dtype=dtype, format=format, fov=fov, pfov=pfov)._image
	frame = imutils.resize(frame, width=RESIZED_WIDTH)
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	kp2, des2 = detector.detectAndCompute(frame_gray, None)
	pair_matches = bf.knnMatch(des2, des1, k=2)
	matches = []
	for m, n in pair_matches:
		if m.distance < 0.7*n.distance:
			matches.append(m)
	matches = sorted(matches, key=lambda x: x.distance)

	if len(matches) < 4:
		print('Not enough matches found!')
		continue

	# Find homography matrix and do perspective transform
	src_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	H, _ = cv.findHomography(src_pts, dst_pts, method=cv.FM_LMEDS, ransacReprojThreshold=REPROJECT_THRESHOLD, confidence=0.99)
	warped_old_gray = cv.warpPerspective(old_gray, H, (frame_gray.shape[1], frame_gray.shape[0]))

	# If there are any black pixels in the warped image, replace them with the corresponding pixels in the new frame
	for i in range(warped_old_gray.shape[0]):
		for j in range(warped_old_gray.shape[1]):
			if warped_old_gray[i, j] <= 0:
				warped_old_gray[i, j] = frame_gray[i, j]

	# Calculate dense optical flow
	# flow = cv.calcOpticalFlowFarneback(warped_old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flow = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(warped_old_gray, frame_gray, None)
	mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang*180/np.pi/2
	hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
	dense_frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

	# Calculate sparse optical flow
	p0 = cv.goodFeaturesToTrack(warped_old_gray, mask = None, **feature_params)
	p1, st, err = cv.calcOpticalFlowPyrLK(warped_old_gray, frame_gray, p0, None, **lk_params)
	if p1 is not None:
		good_old = p0[st==1]
		good_new = p1[st==1]

	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = new.ravel()
		c, d = old.ravel()
		mask_sparse = cv.line(mask_sparse, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
		sparse_frame = cv.circle(np.zeros_like(frame), (int(a), int(b)), 5, color[i].tolist(), -1)

	# Save images
	if DISPLAY_FLOW == "dense": flow_frame = cv.add(frame, dense_frame)
	elif DISPLAY_FLOW == "sparse": flow_frame = cv.add(frame, sparse_frame)
	else: flow_frame = cv.add(frame, cv.add(dense_frame, sparse_frame))
	cv.imwrite(f'.\\output\\{args.images}\\{"{:03d}".format(img_index - 1)}_warped_old.jpg', warped_old_gray)
	cv.imwrite(f'.\\output\\{args.images}\\{"{:03d}".format(img_index - 1)}_frame.jpg', frame_gray)
	cv.imwrite(f'.\\output\\{args.images}\\{"{:03d}".format(img_index - 1)}_flow.jpg', flow_frame)

	# Resize images for viewing
	resized_warped_old = imutils.resize(warped_old_gray, width=RESIZED_WIDTH_VIEWING)
	resized_frame_gray = imutils.resize(frame_gray, width=RESIZED_WIDTH_VIEWING)
	resized_frame = imutils.resize(frame, width=RESIZED_WIDTH_VIEWING)
	resized_old = imutils.resize(old_frame, width=RESIZED_WIDTH_VIEWING)
	resized_flow = imutils.resize(flow_frame, width=RESIZED_WIDTH_VIEWING)

	# Draw the tracks
	matches_img = cv.drawMatches(resized_frame, kp2, resized_old, kp1, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	# Show images side by side
	cv.imshow(
		'frame',
		np.vstack((
			np.hstack((cv.cvtColor(resized_warped_old, cv.COLOR_GRAY2BGR), cv.cvtColor(resized_frame_gray, cv.COLOR_GRAY2BGR))), 
			np.hstack((resized_old, resized_flow)),
			matches_img,
		))
	)
	k = cv.waitKey(0)
	if k == 27:
		break

	# Now update the previous frame and previous points
	old_frame = frame.copy()
	old_gray = frame_gray.copy()
	mask = np.zeros_like(old_gray)
	mask_sparse = np.zeros_like(old_frame)
	kp1 = kp2
	des1 = des2
	p0 = good_new.reshape(-1, 1, 2)
	objpoints = []
	imgpoints = []

cv.destroyAllWindows()