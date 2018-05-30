import numpy as np
import cv2
import os
import sys
import re
import csv
import time
import json

MAX_SPECIES = int(sys.argv[1])
WIDTH = int(sys.argv[2])
HEIGHT = int(sys.argv[3])
 
ELASTICSEARCH_URL = os.environ['ELASTICSEARCH_URL']
ELASTICSEARCH_USER = os.environ['ELASTICSEARCH_USER']
ELASTICSEARCH_PASS = os.environ['ELASTICSEARCH_PASS']
ELASTICSEARCH_INDEX = os.environ['ELASTICSEARCH_INDEX']
ELASTICSEARCH_INDEX_TYPE = os.environ['ELASTICSEARCH_INDEX_TYPE']

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

sp = SimplePreprocessor(WIDTH, HEIGHT)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(found, verbose=500)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
label_10 = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, label_10, test_size=0.25, random_state=42)

print("Received parameters: " + os.environ['PARAMETERS'])
parameters = json.loads(os.environ['PARAMETERS'])

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
params['image_width'] = WIDTH
params['image_height'] = HEIGHT

hyperparameters = params

import itertools
technique = type(model).__name__

experiment_description = ""
experiment_id = ""

if ('BUILD_NAME' in os.environ):
  build_name = os.environ['BUILD_NAME']

if ('BUILD_NUMBER' in os.environ):
  build_number = os.environ['BUILD_NUMBER']

if ('EXPERIMENT_DESCRIPTION' in os.environ):
  experiment_description = os.environ['EXPERIMENT_DESCRIPTION']

if ('EXPERIMENT_ID' in os.environ):
  experiment_id = os.environ['EXPERIMENT_ID']

def replace(x):
    if (x == "avg / total"):
        return "avg"
    else:
        if (x == "f1-score"):
            return "f1"
        else:
            return x

keys = ([
    'build name', 'build number', 'technique', 
    'hyperparameters', 'dataset', 'duration', 
    'experiment_description', 'experiment_id',
    'label', 'precision', 'recall', 'f1-score', 'support'])
data = [[
    build_name, build_number, technique, 
    hyperparameters, dataset, duration,
    experiment_description, experiment_id] + 
    replace(x) for x in report]

objects = [
  dict(zip(keys, values))
  for values in data
]

import urllib.parse
import urllib.request
import requests
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

import urllib.parse
import urllib.request

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

url = ELASTICSEARCH_URL + "/api/console/proxy?path=%2F" + ELASTICSEARCH_INDEX + "%2F" + ELASTICSEARCH_INDEX_TYPE + "%2F&method=POST"

import base64

for o in objects:
  if (not ELASTICSEARCH_USER):
    reply = requests.post(
      url,
      headers={"kbn-xsrf": "reporting"},
      json = o
    )
  else:
    reply = requests.post(
      url,
      headers={"kbn-xsrf": "reporting"},
      auth = requests.auth.HTTPBasicAuth(ELASTICSEARCH_USER, ELASTICSEARCH_PASS),
      json = o
    )

  print("reply: " + reply.text)