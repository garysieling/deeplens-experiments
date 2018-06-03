import requests

import re
import os
import csv

report = """                                precision    recall  f1-score   support

            Acadian_Flycatcher       0.00      0.00      0.00        23
              Acorn_Woodpecker       1.00      0.12      0.21        25
              Zone_tailed_Hawk       0.00      0.00      0.00        17

                   avg / total       0.02      0.01      0.01     11258"""

def typed(i, x):
  if (i == 0):
    return x
  else:
    if i == 5:
      return int(float(x))
    else:
      return float(x)

report = [
  [typed(i, v) for i, v in enumerate(re.compile(" [ ]+").split(x.strip()))]
  for x in report.split("\n")[2:] 
  if len(x) > 0
]

build_name = 'fast'
build_number = '1'
technique = 'kmeans'
dataset = 'birdsnap'
hyperparameters = {
  'species': 3,
  'neighbors': 10,
  'jobs': 5
}

duration = 7

if ('BUILD_NUMBER' in os.environ):
  build_number = os.environ['BUILD_NUMBER']

keys = (['build name', 'build number', 'technique', 'hyperparameters', 'dataset', 'duration', 'label', 'precision', 'recall', 'f1-score', 'support'])
data = [[build_name, build_number, technique, hyperparameters, dataset, duration] + x for x in report]

import json
objects = [
  dict(zip(keys, values))
  for values in data
]

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

index = "experiments"
index_type = "1"

url = "https://bd333f63f75f41a6af393bc0e0dc6b63.us-east-1.aws.found.io:9243/api/console/proxy?path=%2F" + index + "%2F" + index_type + "%2F&method=POST"

USERNAME = "elastic"
PASSWORD = "C3RPvCScJh8ZQhDsflC5qeVC"

import base64

for o in objects:
  reply = requests.post(
    url,
    headers={"kbn-xsrf": "reporting"},
    auth = requests.auth.HTTPBasicAuth(USERNAME, PASSWORD),
    json = o
  )

  print("reply: " + reply.text)