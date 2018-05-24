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

if ('BUILD_NUMBER' in os.environ):
  build_number = os.environ['BUILD_NUMBER']

keys = (['build name', 'build number', 'technique',  'label', 'precision', 'recall', 'f1-score', 'support'])
data = [[build_name, build_number, technique] + x for x in report]

import json
objects = [
  json.dumps(dict(zip(keys, values)))
  for values in data
]

print(objects)

import splunklib.client as client
splunkargs = {}

service = client.connect(
  host='prd-p-vtk8vp5x5ggv.cloud.splunk.com',
  port='8089',
  username='admin',
  password='changeme',
  scheme='https',
  version='5.0'
)
#cn = service.indexes[index].attach(**kwargs_submit)
