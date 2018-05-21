import numpy as np
import cv2
import os
import sys

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
            if (len(distinct_labels) >= int(sys.argv[1])):
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

todo = SimpleDatasetLoader(preprocessors=[SimplePreprocessor(200, 200)])

from os import listdir
mypath = "/data/birdsnap/download/images"
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

jobs = 5
neighbors = 10

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

