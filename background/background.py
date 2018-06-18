
import cv2


# import the necessary packages
import argparse
import datetime
import imutils
import time

from threading import Thread
import sys
from queue import Queue
import numpy as np

import matplotlib.pyplot as plt

import mxnet as mx
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	r = 1.0
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)

	else:
		r = width / float(w)
		dim = (width, int(h * r))

	resized = cv2.resize(image, dim, interpolation=inter)

	# return the resized image
	return (r, resized)

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

scale = 1

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-min", "--min-area", type=int, default=int(100 * scale * scale), help="minimum area size")
ap.add_argument("-max", "--max-area", type=int, default=int(15000 * scale * scale), help="minimum area size")
ap.add_argument("-maxWidth", "--max-width", type=int, default=int(500 * scale), help="minimum area size")
ap.add_argument("-maxHeight", "--max-height", type=int, default=int(500 * scale), help="minimum area size")

ap.add_argument("-left", "--left", type=int, default=int(400 * scale), help="minimum area size")
ap.add_argument("-right", "--right", type=int, default=int(800 * scale), help="minimum area size")
ap.add_argument("-top", "--top", type=int, default=int(250 * scale), help="minimum area size")
ap.add_argument("-bottom", "--bottom", type=int, default=int(600 * scale), help="minimum area size")

args = vars(ap.parse_args())
 
# initialize the first frame in the video stream
# TODO: I think this needs a calibration mode - first few seconds
firstFrame = None

print("[INFO] starting video file thread...")
fvs = FileVideoStream('http://192.168.1.187:8090/feed.mjpeg').start()

frameNumber = 0
# loop over the frames of the video

start = time.clock()

lastSighting = None

net = get_model('cifar_resnet110_v1', classes=10, pretrained=True)

transform_fn = transforms.Compose([
	transforms.Resize(32),
	transforms.CenterCrop(32),
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

while True:
	if (not fvs.more):
		continue

	frameNumber = frameNumber + 1
	src = fvs.read()

	(imageScale, frame) = resize(src, width=int(1200 * scale))
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)	

	blur = int(41 * scale)
	if (blur % 2 == 0):
		blur = blur + 1

	gray = cv2.GaussianBlur(gray, (blur, blur), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
 
	thresh = cv2.dilate(thresh, None, iterations=2)
	(image, contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
	glitched = False

	chosen = None

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
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
		chosen = (x, y, w, h)

		(xSmall, ySmall, wSmall, hSmall) = chosen
		print(chosen)

		# what were these in the original?
		xBig = round(xSmall / imageScale)
		yBig = round(ySmall / imageScale)
		wBig = round(wSmall / imageScale)
		hBig = round(hSmall / imageScale)
	
		cropped = src[yBig:yBig+hBig, xBig:xBig+wBig]
		#print("A: " + str(cropped.shape))
		typed = mx.nd.array(cropped)
		#print("B: " + str(typed.shape))
		transformed = transform_fn(typed)
		#print("C: " + str(transformed.shape))

		tryMe = mx.nd.expand_dims(
				transformed, 
				axis=0)

		#print(tryMe.shape)

		pred = net(tryMe)
		#print("pred")
		#print(pred)

		# cv2 to mxnet image

		# _cvimresize
		# Argument src must have NDArray type, but got
		#rightType = mxnet.image.imdecode(tryMe)
		#https://github.com/apache/incubator-mxnet/issues/8569
		# conda remove opencv3
		#reshaped = mx.nd.array(transformed)
		#print(reshaped)

		#perative_utils.h:108: Check failed: infershape[attrs.op](attrs, &in_shapes, &out_shapes)
		#
		class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
		
		ind = mx.nd.argmax(pred, axis=1).astype('int')
		print('The input picture is classified as [%s], with probability %.3f.'%
				(class_names[ind.asscalar()], mx.nd.softmax(pred)[0][ind].asscalar()))


	if ((frameNumber % 25) == 0):
		firstFrame = gray

	if (glitched):
		continue

	if (chosen != None):
		(xSmall, ySmall, wSmall, hSmall) = chosen
		print(chosen)

		# what were these in the original?
		xBig = round(xSmall / imageScale)
		yBig = round(ySmall / imageScale)
		wBig = round(wSmall / imageScale)
		hBig = round(hSmall / imageScale)
	
		frame[0:hBig, 0:wBig] = src[yBig:yBig+hBig, xBig:xBig+wBig]

		#small_image.copyTo(frame(x, y, w, h))
		#crop_img = frame[y:y+h, x:x+w]
		#cv2.seamlessClone(crop_img, frame, src_mask, (0, 0), cv2.NORMAL_CLONE)


	cv2.rectangle(frame, (args["left"], args["top"]), (args["right"], args["bottom"]), (255, 0, 0), 2)

	# draw the text and timestamp on the frame
	#cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	end = time.clock()

	cv2.putText(frame, str(frameNumber) + " / " + str( frameNumber / (end - start) ),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# coupe issues- 
	  # need periodic recalibration
		# flicker - can't have over certain percent

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	#cv2.imshow("Thresh", thresh)
	#cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
 
#camera.release()
cv2.destroyAllWindows()
