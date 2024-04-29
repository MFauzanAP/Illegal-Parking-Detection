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
old_frame = imutils.resize(old_frame, width=600)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255

while(1):
	ret, frame = cap.read()
	if not ret:
		print('No frames grabbed!')
		break

	frame = imutils.resize(frame, width=600)
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	# calculate optical flow
	p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	if p1 is not None:
		good_new = p1[st==1]
		good_old = p0[st==1]

	# Find homography matrix and do perspective transform
	H, _ = cv.findHomography(good_old, good_new, cv.RANSAC, 5.0)
	warped_old_gray = cv.warpPerspective(old_gray, H, (old_gray.shape[1], old_gray.shape[0]))

	# # draw the tracks
	# for i, (new, old) in enumerate(zip(good_new, good_old)):
	# 	a, b = new.ravel()
	# 	c, d = old.ravel()
	# 	mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
	# 	warped_frame = cv.circle(warped_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
	# img = cv.add(warped_frame, mask)

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
			np.hstack((old_frame, cv.add(old_frame, bgr)))
		))
	)
	k = cv.waitKey(0)
	if k == 27:
		break

	# Now update the previous frame and previous points
	old_frame = frame.copy()
	old_gray = frame_gray.copy()
	mask = np.zeros_like(old_frame)
	p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()