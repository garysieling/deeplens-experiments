import urllib.parse
import urllib.request

import logging
import contextlib
import re
import os

import urllib.parse
import urllib.request
import requests
import base64
import uuid

ELASTICSEARCH_URL = os.environ['ELASTICSEARCH_URL']
ELASTICSEARCH_USER = os.environ['ELASTICSEARCH_USER']
ELASTICSEARCH_PASS = os.environ['ELASTICSEARCH_PASS']
ELASTICSEARCH_INDEX = os.environ['ELASTICSEARCH_INDEX']
ELASTICSEARCH_INDEX_TYPE = os.environ['ELASTICSEARCH_INDEX_TYPE']
GIT_SHA = os.environ['GIT_SHA']

def replace(x):
  if (x == "avg / total"):
    return "avg"
  else:
    if (x == "f1-score"):
      return "f1"
    else:
      return x

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

def typed(i, x):
  if (i == 0):
    return x
  else:
    if i == 5:
      return int(float(x))
    else:
      return float(x)

def save_report(report, start_time, end_time, model, species, width, height, dataset): 
  report = [
    [typed(i, v) for i, v in enumerate(re.compile(" [ ]+").split(x.strip()))]
    for x in report.split("\n")[2:] 
    if len(x) > 0
  ]

  build_name = ''
  build_number = ''
  duration = end_time - start_time

  params = model.get_params()
  params['species'] = species
  params['image_width'] = width
  params['image_height'] = height
  params['git_sha'] = GIT_SHA

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
 
  runid = uuid.uuid4().hex

  keys = ([
    'runid', 'build name', 'build number', 'technique', 
    'hyperparameters', 'dataset', 'duration', 
    'experiment_description', 'experiment_id',
    'label', 'precision', 'recall', 'f1-score', 'support'])

  data = [[
    runid, build_name, build_number, technique, 
    hyperparameters, dataset, duration,
    experiment_description, experiment_id] + 
    replace(x) for x in report]

  objects = [
    dict(zip(keys, values))
    for values in data
  ]

  debug_requests_on()

  logging.basicConfig(level=logging.DEBUG)

  url = ELASTICSEARCH_URL + "%2F" + ELASTICSEARCH_INDEX + "%2F" + ELASTICSEARCH_INDEX_TYPE + "%2F&method=POST"

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