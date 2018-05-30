import os
import sys
import re
import csv
import time
import json

from loader import *

MAX_SPECIES = int(sys.argv[1])
WIDTH = int(sys.argv[2])
HEIGHT = int(sys.argv[3])
DATASET = sys.argv[4]

start_time = time.clock() * 1000

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

sp = SimplePreprocessor(WIDTH, HEIGHT)
sdl = SimpleDatasetLoader(dataset = dataset, species = MAX_SPECIES, preprocessors=[sp])
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

report = classification_report(testY, model.predict(testX), target_names=le.classes_)

import save
save_report(report, start_time, end_time, model, MAX_SPECIES, WIDTH, HEIGHT)
