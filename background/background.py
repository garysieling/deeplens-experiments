# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
from threading import Thread
import sys
from queue import Queue

class FileVideoStream:
	def __init__(self, path, queueSize=128):
		self.stream = cv2.VideoCapture(path)
		self.stopped = False

		self.Q = Queue(maxsize=queueSize)

	def start(self):
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return

			if not self.Q.full():
				(grabbed, frame) = self.stream.read()

				if not grabbed:
					self.stop()
					return

				self.Q.put(frame)

	def read(self):
		return self.Q.get()

	def more(self):
		return self.Q.qsize() > 0

	def stop(self):
		self.stopped = True

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-min", "--min-area", type=int, default=100, help="minimum area size")
ap.add_argument("-max", "--max-area", type=int, default=15000, help="minimum area size")
ap.add_argument("-maxWidth", "--max-width", type=int, default=500, help="minimum area size")
ap.add_argument("-maxHeight", "--max-height", type=int, default=500, help="minimum area size")


ap.add_argument("-left", "--left", type=int, default=400, help="minimum area size")
ap.add_argument("-right", "--right", type=int, default=800, help="minimum area size")
ap.add_argument("-top", "--top", type=int, default=250, help="minimum area size")
ap.add_argument("-bottom", "--bottom", type=int, default=600, help="minimum area size")

args = vars(ap.parse_args())
 
# initialize the first frame in the video stream
# TODO: I think this needs a calibration mode - first few seconds
firstFrame = None

print("[INFO] starting video file thread...")
fvs = FileVideoStream('http://192.168.1.187:8090/feed.mjpeg').start()

frameNumber = 0
# loop over the frames of the video
while True:
	if (not fvs.more):
		continue

	frameNumber = frameNumber + 1
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = fvs.read()
	text = "Unoccupied"

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=1200)

	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)	


	#gray = cv2.GaussianBlur(gray, (21, 21), 0)
	gray = cv2.GaussianBlur(gray, (41, 41), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(image, contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
 
	# loop over the contours
	glitched = False

	for c in contours:
		x,y,w,h = cv2.boundingRect(c)

		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
 
		if cv2.contourArea(c) > args["max_area"]:
			glitched = True
			continue
 
		if w > args["max_width"]:
			continue

		if h > args["max_height"]:
			continue

		if x < args["left"]:
			continue

		if (x+w) > args["right"]:
			continue

		if y < args["top"]:
			continue

		if (y+h) > args["bottom"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		text = "Occupied"

	if ((frameNumber % 25) == 0):
		firstFrame = gray

	#if (glitched):
	#	continue

	cv2.rectangle(frame, (args["left"], args["top"]), (args["right"], args["bottom"]), (255, 0, 0), 2)

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, str(frameNumber),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# coupe issues- 
	  # need periodic recalibration
		# flicker - can't have over certain percent

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
 
#camera.release()
cv2.destroyAllWindows()
