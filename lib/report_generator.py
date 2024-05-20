import os
import json
import humanize
import cv2 as cv
import numpy as np

from fpdf import FPDF, Align, FontFace
from datetime import datetime, timedelta

class ReportGenerator(FPDF):
	WHITE = 255
	LIGHT_GREY = 245
	GREY = 220
	def __init__(self, dataset, img_width, car_detection, motion_detection, geojson_plotter, ipc_detection):
		super().__init__()

		self.dataset = dataset
		self.img_width = img_width
		self.car_detection = car_detection
		self.motion_detection = motion_detection
		self.geojson_plotter = geojson_plotter
		self.ipc_detection = ipc_detection

		self.set_title("Illegal Parking Detection Report")
		self.set_creation_date(datetime.now())

		cwd = os.getcwd()
		self.num_images = len([name for name in os.listdir(f"{cwd}\\data\\{dataset}") if os.path.isfile(os.path.join(f"{cwd}\\data\\{dataset}", name)) and name[-4:] == ".jpg"])

		# Keep track of how long the pipeline takes
		self.start_time = cv.getTickCount()

	# Generate and export the final report
	def export(self):

		# Calculate the time taken to process the dataset
		self.time_taken = (cv.getTickCount() - self.start_time) / cv.getTickFrequency()
		print(f'Time taken: {self.time_taken} seconds')

		# Generate the report
		self.add_page()
		self.title_date()
		self.report_overview()
		self.ipc_table()
		self.mission_details()
		self.camera_details()

		# Export the report to a PDF file
		self.output(f'.\\output\\{self.dataset}\\report.pdf')

	def header(self):
		self.image(name=".\\docs\\imgs\\qu_main_logo.png", x=Align.R, w=33)
		self.set_font("helvetica", size=8)
		self.set_y(10)
		self.cell(w=0, h=10, text="MECH420 - Introduction to Drones", align="L")
		self.ln(20)

	def footer(self):
		self.set_font('helvetica', 'I', 8)
		self.set_y(-15)
		self.cell(w=0, text="Group 8", align="L")
		self.set_y(-15)
		self.cell(w=0, text=f'{self.page_no()}', align="C")
		self.set_y(-15)
		self.cell(w=0, text="GitHub Source Code", link="https://github.com/MFauzanAP/Illegal-Parking-Detection", align="R")

	def title_date(self):
		self.set_font("helvetica", "B", size=16)
		self.cell(w=0, h=10, text=self.title, align="C")
		self.ln(8)
		self.set_font("helvetica", "I", size=10)
		self.cell(w=0, h=10, text=f"Processed on {self.creation_date.strftime('%Y-%m-%d %H:%M:%S')}", align="C")
		self.ln(12)

	def report_overview(self):
		# Add screenshot of mission route
		if os.path.exists(f".\\output\\{self.dataset}\\mission-path\\path.png"):
			self.image(name=f".\\output\\{self.dataset}\\mission-path\\path.png", w=160, x=Align.C)
		else:
			self.set_fill_color(self.LIGHT_GREY)
			self.set_x((self.w / 2) - 60)
			self.cell(w=120, h=60, text="No mission route available", align="C", fill=True)
			self.set_fill_color(self.WHITE)
			self.ln(60)

		# Add the section title
		self.ln(12)
		self.set_font("helvetica", "B", 14)
		self.cell(w=0, h=10, text="Report Overview", align="C")
		self.ln(12)

		# Create table with general info about the processing
		self.set_font("helvetica", "", 11)
		with self.table(first_row_as_headings=False, cell_fill_color=self.LIGHT_GREY, cell_fill_mode="EVEN_COLUMNS") as t:
			processing_time = humanize.precisedelta(timedelta(seconds=self.time_taken))
			mission_duration = humanize.precisedelta(self.geojson_plotter.mission_end_time - self.geojson_plotter.mission_start_time)

			t.row(["Dataset", self.dataset])
			t.row(["Number of Illegally Parked Cars", f"{self.ipc_detection.num_ipcs} detected"])
			t.row(["Number of Images", f"{self.num_images} processed"])
			t.row(["Processing Time", processing_time])
			t.row(["Mission Duration", mission_duration])

		self.add_page()

	def ipc_table(self):
		self.ln(12)
		self.set_font("helvetica", "B", 14)
		self.cell(w=0, h=10, text="Illegally Parked Cars", align="C")
		self.ln(12)

		# Add screenshot of the clustered IPCs point cloud
		if os.path.exists(f".\\output\\{self.dataset}\\clustered-point-cloud\\point_cloud.png"):
			self.image(name=f".\\output\\{self.dataset}\\clustered-point-cloud\\point_cloud.png", w=120, x=Align.C)
		else:
			self.set_fill_color(self.LIGHT_GREY)
			self.set_x((self.w / 2) - 60)
			self.cell(w=120, h=60, text="No point cloud data available", align="C", fill=True)
			self.set_fill_color(self.WHITE)
			self.ln(60)

		# Create table of all tagged IPCs
		self.ln(12)
		self.set_font("helvetica", "", 10)
		if self.ipc_detection.num_ipcs > 0:
			headings_style = FontFace(fill_color=self.GREY)
			col_widths = (35, 45, 30, 30, 35)
			with self.table(
				headings_style=headings_style,
				col_widths=col_widths,
				text_align="CENTER",
				cell_fill_color=self.LIGHT_GREY,
				cell_fill_mode="EVEN_ROWS",
				repeat_headings=0
			) as t:
				t.row(["Screenshot", "Location", "Timestamp", "Time Parked", "# of Detections"])
				# for ipc in ipcs:
				for ipc in self.ipc_detection.tagged_ipcs:
					row = t.row()
					row.cell(img=ipc["img"])
					row.cell(f"{ipc["lat"]}, {ipc["lon"]}", link=ipc["maps"])
					row.cell(datetime.fromtimestamp(ipc["time"]).strftime("%Y-%m-%d %H:%M:%S"))
					row.cell(humanize.precisedelta(ipc["parking_time"]))
					row.cell(f"{ipc["num_detections"]} / {self.num_images} images ({ipc["num_detections"] / self.num_images:.2%})")
		else:
			self.set_fill_color(self.LIGHT_GREY)
			self.cell(w=0, h=40, text="No illegally parked cars detected", align="C", fill=True)
			self.set_fill_color(self.WHITE)
			self.ln(40)

		# Add screenshot of the potential IPCs point cloud
		self.ln(12)
		if os.path.exists(f".\\output\\{self.dataset}\\point-cloud-histogram\\potential_ipcs.png"):
			self.image(name=f".\\output\\{self.dataset}\\point-cloud-histogram\\potential_ipcs.png", w=120, x=Align.C)
		else:
			self.set_fill_color(self.LIGHT_GREY)
			self.set_x((self.w / 2) - 60)
			self.cell(w=120, h=60, text="No potential IPC histogram available", align="C", fill=True)
			self.set_fill_color(self.WHITE)
			self.ln(60)

	def mission_details(self):
		self.ln(12)
		self.set_font("helvetica", "B", 14)
		self.cell(w=0, h=10, text="Mission Details", align="C")
		self.ln(12)

		# Create table with general info about the mission
		self.set_font("helvetica", "", 11)
		with self.table(first_row_as_headings=False, cell_fill_color=self.LIGHT_GREY, cell_fill_mode="EVEN_COLUMNS") as t:
			distance_travelled = sum([ self.geojson_plotter.calculate_haversine(p, self.geojson_plotter.capture_points[i+1]) for i, p in enumerate(self.geojson_plotter.capture_points[:-1]) ])
			min_alt = min(self.geojson_plotter.altitudes)
			max_alt = max(self.geojson_plotter.altitudes)
			mission_start = datetime.fromtimestamp(self.geojson_plotter.mission_start_time)
			mission_end = datetime.fromtimestamp(self.geojson_plotter.mission_end_time)
			mission_duration = humanize.precisedelta(mission_end - mission_start)

			t.row(["Distance Travelled", f"{distance_travelled:.2f} meters"])
			t.row(["Relative Altitude Range", f"{min_alt:.2f} to {max_alt:.2f} meters above ground"])
			t.row(["Mission Start Time", mission_start.strftime("%Y-%m-%d %H:%M:%S")])
			t.row(["Mission End Time", mission_end.strftime("%Y-%m-%d %H:%M:%S")])
			t.row(["Mission Duration", mission_duration])

	def camera_details(self):
		self.ln(12)
		self.set_font("helvetica", "B", 14)
		self.cell(w=0, h=10, text="Camera Details", align="C")
		self.ln(12)

		# Create table with general info about the camera
		self.set_font("helvetica", "", 11)
		with self.table(first_row_as_headings=False, cell_fill_color=self.LIGHT_GREY, cell_fill_mode="EVEN_COLUMNS") as t:
			t.row(["Image Resolution", f"{self.geojson_plotter.image_resolution} pixels"])
			t.row(["Focal Length", f"{self.geojson_plotter.focal_length:.2f} mm"])
			t.row(["Mega Pixels", f"{self.geojson_plotter.mega_pixels} MP"])
			t.row(["Capture Rate", f"{self.geojson_plotter.capture_rate} images per second"])
			t.row(["Field of View", self.geojson_plotter.field_of_view])
