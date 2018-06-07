import cv2
import os
import numpy as np
from os.path import isfile, join
from os import listdir

class SimplePreprocessor:
  def __init__(self, width, height, grayscale, inter=cv2.INTER_AREA):
    self.width = width
    self.height = height
    self.inter = inter
    self.grayscale = grayscale
    
  def preprocess(self, image):
    processed = cv2.resize(image, (self.width, self.height), interpolation=self.inter)

    if (self.grayscale):
      processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    return processed


class SimpleDatasetLoader:
  def __init__(self, dataset, species, path, preprocessors=None):
    self.preprocessors = preprocessors
    self.species = species
    self.dataset = dataset
    self.path = path

    if self.preprocessors is None:
      self.preprocessors = []
            
  def load(self, verbose=-1):
    print("loading")
    
    imagePaths = []
    for root, subFolders, files in os.walk(self.path):
      print(files)
      
      for folder in subFolders:
        for f in listdir(self.path + "/" + folder):
          imagePaths.append(self.path + "/" + folder + "/" + f)

    data = []
    labels = []
    distinct_labels = []

    for (i, imagePath) in enumerate(imagePaths):
      if (len(distinct_labels) > self.species):
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
