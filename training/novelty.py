import os
import sys
import re
import csv
import time
import json
import mxnet as mx
from loader import *

WIDTH = int(sys.argv[1])
HEIGHT = int(sys.argv[2])
DATASET = "birdsnap"

start_time = time.clock() * 1000

sp = SimplePreprocessor(WIDTH, HEIGHT, True)
sdl = SimpleDatasetLoader(
  dataset = DATASET, 
  species = -1,
  path = '/frames',
  preprocessors=[sp])

(data, labels) = sdl.load(verbose=500)
print(data)

