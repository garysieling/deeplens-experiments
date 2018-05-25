import re
import os
import csv

report = """                                precision    recall  f1-score   support

            Acadian_Flycatcher       0.00      0.00      0.00        23
              Acorn_Woodpecker       1.00      0.12      0.21        25
              Zone_tailed_Hawk       0.00      0.00      0.00        17

                   avg / total       0.02      0.01      0.01     11258"""

report = [
  re.compile(" [ ]+").split(x.strip())
  for x in report.split("\n")[2:] 
  if len(x) > 0
]

build_name = 'fast'
build_number = '1'
technique = 'kmeans'
dataset = 'birdsnap'
hyperparameters = 'species=3,neighbors=10,jobs=5'
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

#curl -k  https://input-prd-p-vtk8vp5x5ggv.cloud.splunk.com:8088/services/collector/event -H "Authorization: Splunk c8b8b9fd-f366-4c6f-9f17-993cae466d58" -d '{"event": "hello world"}'

# index=""
# source=""
