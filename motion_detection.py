import numpy as np
import cv2 as cv
import argparse
import imutils

parser = argparse.ArgumentParser(
    description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
	The example file can be downloaded from: \
	https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4'
)
parser.add_argument('images', type=str, help='path to image files', default="\\data")
args = parser.parse_args()

cap = cv.VideoCapture(f"{args.images}\\%5d.jpg")
detector = cv.SIFT_create(700)
bf = cv.BFMatcher()

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
old_frame = imutils.resize(old_frame, width=400)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
kp1, des1 = detector.detectAndCompute(old_gray, None)
H_old = np.eye(3)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)

hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255

while(1):
	ret, frame = cap.read()
	if not ret:
		print('No frames grabbed!')
		break

	frame = imutils.resize(frame, width=400)
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	kp2, des2 = detector.detectAndCompute(frame_gray, None)
	pair_matches = bf.knnMatch(des2, des1, k=2)
	matches = []
	for m, n in pair_matches:
		if m.distance < 0.7*n.distance:
			matches.append(m)
	matches = sorted(matches, key=lambda x: x.distance)
	matches = matches[:min(len(matches), 20)]

	if len(matches) < 4:
		print('Not enough matches found!')
		continue

	# Find homography matrix and do perspective transform
	src_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
	H = np.matmul(H_old, H)
	warped_old_gray = cv.warpPerspective(old_gray, H, (frame_gray.shape[1], frame_gray.shape[0]))
	
	# If there are any black pixels in the warped image, replace them with the corresponding pixels in the new frame
	for i in range(warped_old_gray.shape[0]):
		for j in range(warped_old_gray.shape[1]):
			if warped_old_gray[i, j] == 0:
				warped_old_gray[i, j] = frame_gray[i, j]

	# draw the tracks
	matches_img = cv.drawMatches(frame, kp2, old_frame, kp1, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	# Calculate dense optical flow
	flow = cv.calcOpticalFlowFarneback(warped_old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang*180/np.pi/2
	hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
	bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

	# Show images side by side
	cv.imshow(
		'frame',
		np.vstack((
			np.hstack((cv.cvtColor(warped_old_gray, cv.COLOR_GRAY2BGR), frame)), 
			np.hstack((old_frame, cv.add(old_frame, bgr))),
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
	kp1 = kp2
	des1 = des2

cv.destroyAllWindows()