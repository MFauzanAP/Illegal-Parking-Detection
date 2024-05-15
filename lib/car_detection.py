import os
import json
import imutils
import contextlib
import cv2 as cv

from lib import DarkHelp

class CarDetection():
	DETECTION_THRESHOLD = 0.2
	TILE_SIZE = 360
	CONFIG_PATH = "model/model.cfg".encode("utf-8")
	NAMES_PATH = "model/model.names".encode("utf-8")
	WEIGHTS_PATH = "model/model_best.weights".encode("utf-8")

	def __init__(self, dataset, img_width, output_cars=True):
		self.dataset = dataset
		self.img_width = img_width
		self.output_cars = output_cars

		# Initialize DarkHelp with the neural network files.
		with open(os.devnull, "w") as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
			self.dh = DarkHelp.CreateDarkHelpNN(self.CONFIG_PATH, self.NAMES_PATH, self.WEIGHTS_PATH)
			if not self.dh:
				print("""
					Failed to allocate a DarkHelp object.  Possible problems include:

					1) missing neural network files, or files are in a different directory
					2) libraries needed by DarkHelp or Darknet have not been installed
					3) errors in DarkHelp or Darknet libraries
				""")

			# Configure DarkHelp to ensure it behaves like we want.
			DarkHelp.SetThreshold(self.dh, self.DETECTION_THRESHOLD)
			DarkHelp.EnableTiles(self.dh, False)
			DarkHelp.EnableSnapping(self.dh, True)
			DarkHelp.EnableUseFastImageResize(self.dh, False)
			DarkHelp.EnableNamesIncludePercentage(self.dh, True)
			DarkHelp.EnableAnnotationAutoHideLabels(self.dh, False)
			DarkHelp.EnableAnnotationIncludeDuration(self.dh, False)
			DarkHelp.EnableAnnotationIncludeTimestamp(self.dh, False)
			DarkHelp.SetAnnotationLineThickness(self.dh, 1)

	# Call to destroy the DarkHelp object
	def destroy(self): DarkHelp.DestroyDarkHelpNN(self.dh)

	# Shortcut for getting the path to a directory
	def get_path(self, dir, ext="jpg"): return f".\\output\\{self.dataset}\\{dir}\\{"{:05d}".format(self.img_index)}.{ext}"

	# Analyze the image
	def analyze(self, img, img_index):
		self.img = img
		self.gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
		self.img_index = img_index

		# Resize the image and tile it in a 2x2 grid for better detection
		self.grid_resize()

		# Detect cars in the grid image and save the prediction results
		self.detect_cars()

	# Resize the image and tile it in a 2x2 grid for better detection
	def grid_resize(self):
		self.resized = imutils.resize(self.img, height=self.TILE_SIZE)

		# Create a 2x2 grid of the resized image
		grid = cv.vconcat([
			cv.hconcat([self.resized, self.resized]),
			cv.hconcat([self.resized, self.resized]),
		])

		# Save the grid image
		cv.imwrite(self.get_path("grid-resized"), grid)

	# Detect cars in the grid image and save the prediction results
	def detect_cars(self):
		DarkHelp.PredictFN(self.dh, self.get_path("grid-resized").encode("utf-8"))

		# Save the annotated image before processing the prediction results
		DarkHelp.Annotate(self.dh, self.get_path("cars").encode("utf-8"))

		# Process the prediction results by only taking predictions from the first tile and scaling the rect
		results = json.loads(DarkHelp.GetPredictionResults(self.dh).decode())
		self.predictions = results["file"][0]["prediction"]
		bboxes = []
		for prediction in self.predictions:
			rect_x = prediction["rect"]["x"]
			rect_y = prediction["rect"]["y"]
			rect_w = prediction["rect"]["width"]
			rect_h = prediction["rect"]["height"]

			# Skip any detections that are outside the bounds of the first tile
			bottom_right = (rect_x + rect_w, rect_y + rect_h)
			if bottom_right[0] > self.resized.shape[1] or bottom_right[1] > self.resized.shape[0]: continue

			# Calculate scaling factor to account for grid resizing and scaling to the original image size
			scale_factor = (self.img.shape[0] / self.resized.shape[0])

			# Multiply the rect coordinates by the scale factor
			rect_x = int(rect_x * scale_factor)
			rect_y = int(rect_y * scale_factor)
			rect_w = int(rect_w * scale_factor)
			rect_h = int(rect_h * scale_factor)

			# Append the bounding box to the list
			bboxes.append(((rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h)))

		# Save the prediction results to a JSON file
		with open(self.get_path("cars-json", "json"), "w") as f:
			json.dump(self.predictions, f, indent=4)

		# Draw the annotations on the image and save it
		self.img_annotation = self.img.copy()
		for bbox in bboxes:
			top_left, bottom_right = bbox
			cv.rectangle(self.img_annotation, top_left, bottom_right, (0, 0, 255), 4)
		cv.imwrite(self.get_path("cars-bbox"), self.img_annotation)
