import os
import sys
import re
import csv
import time
import json

from loader import *

WIDTH = int(sys.argv[1])
HEIGHT = int(sys.argv[2])
DATASET = "birdsnap"

start_time = time.clock() * 1000

sp = SimplePreprocessor(WIDTH, HEIGHT, True)
sdl = SimpleDatasetLoader(
  dataset = DATASET, 
  species = -1,
  preprocessors=[sp])
#(data, labels) = sdl.load(verbose=500)
#data = data.reshape((data.shape[0], 3 * WIDTH * HEIGHT))

#print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

#le = LabelEncoder()
#label_10 = le.fit_transform(labels)

#(trainX, testX, trainY, testY) = train_test_split(data, label_10, test_size=0.25, random_state=42)

#print("Received parameters: " + os.environ['PARAMETERS'])
#parameters = json.loads(os.environ['PARAMETERS'])

#print("[INFO] evaluating k-NN classifier...")
#model = KNeighborsClassifier(**parameters)
#model.fit(trainX, trainY)

#report = classification_report(testY, model.predict(testX), target_names=le.classes_)

#from save import save_report
#save_report(report, start_time, model, MAX_SPECIES, WIDTH, HEIGHT, DATASET)
