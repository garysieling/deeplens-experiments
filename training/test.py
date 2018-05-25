import numpy as np
import cv2
import os
import sys
import re
import csv
import time
import json

MAX_SPECIES = int(sys.argv[1])

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagePaths, verbose=-1):
        print("loading")
        data = []
        labels = []
        distinct_labels = []

        for (i, imagePath) in enumerate(imagePaths):
            if (len(distinct_labels) >= MAX_SPECIES):
              continue

            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if (not label in distinct_labels):
              distinct_labels.append(label)
            
            if (image is None):
                print("Broken image " + imagePath)
                continue
            
            height, width, channels = image.shape
            
            if (height == 0):
                continue
            if (width == 0):
                continue
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

                    data.append(image)
                    labels.append(label)
                    
                    if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                        print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
                        
        return (np.array(data), np.array(labels))

start_time = time.clock() * 1000

todo = SimpleDatasetLoader(preprocessors=[SimplePreprocessor(200, 200)])

from os import listdir

dataset = 'birdsnap'
mypath = '/data/' + dataset + '/download/images'
from os.path import isfile, join

found = []
for root, subFolders, files in os.walk(mypath):
    for folder in subFolders:
        for f in listdir(mypath + "/" + folder):
            found.append(mypath + "/" + folder + "/" + f)        

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import argparse

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(found, verbose=500)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
label_10 = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, label_10, test_size=0.25, random_state=42)

class Parameters(object):
    def __init__(self, data):
        self.__dict__ = json.loads(data)

parameters = Parameters(os.environ['PARAMETERS'])
print("Received parameters: " + os.environ['PARAMETERS'])

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(**parameters)
model.fit(trainX, trainY)

end_time = time.clock() * 1000

report=classification_report(testY, model.predict(testX), target_names=le.classes_)

report = [
  re.compile(" [ ]+").split(x.strip())
  for x in report.split("\n")[2:] 
  if len(x) > 0
]

build_name = ''
build_number = ''
duration = end_time - start_time

params = model.get_params()
params['species'] = MAX_SPECIES
hyperparameters = json.dumps(params)

import itertools
technique = type(model).__name__

if ('BUILD_NAME' in os.environ):
  build_name = os.environ['BUILD_NAME']

if ('BUILD_NUMBER' in os.environ):
  build_number = os.environ['BUILD_NUMBER']

if ('BUILD_NUMBER' in os.environ):
  build_number = os.environ['BUILD_NUMBER']

keys = (['build name', 'build number', 'technique', 'hyperparameters', 'dataset', 'duration', 'label', 'precision', 'recall', 'f1-score', 'support'])
data = [[build_name, build_number, technique, hyperparameters, dataset, duration] + x for x in report]

objects = [
  dict(zip(keys, values))
  for values in data
]

import splunklib.client as client
splunkargs = {}

token="c8b8b9fd-f366-4c6f-9f17-993cae466d58"
port='8088'
import urllib.parse
import urllib.request

host = "input-prd-p-vtk8vp5x5ggv.cloud.splunk.com"

url = "https://" + host + ":8088/services/collector/event"

import requests
headers = {
  'Authorization': 'Splunk ' + token,
  'X-Splunk-Request-Channel': 'dea42704-9036-428b-a319-0025754046ec'
}

import logging
import contextlib
try:
    from http.client import HTTPConnection # py3
except ImportError:
    from httplib import HTTPConnection # py2

def debug_requests_on():
    '''Switches on logging of the requests module.'''
    HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

debug_requests_on()

logging.basicConfig(level=logging.DEBUG)

print(objects[0])

for o in objects:
  jsonData = {
    "host":"jenkins",
    "index":"model-performance",
    "sourcetype":"http",
    "source":"http:python-report",
    "event": json.dumps(o)
  }
  r = requests.post(url, json=jsonData, headers=headers, verify=False)
  print(r.json())