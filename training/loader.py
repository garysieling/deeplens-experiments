import cv2
import os
import numpy as np
from os.path import isfile, join
from os import listdir

class SimplePreprocessor:
  def __init__(self, width, height, inter=cv2.INTER_AREA):
    self.width = width
    self.height = height
    self.inter = inter
    
  def preprocess(self, image):
    return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class SimpleDatasetLoader:
  def __init__(self, dataset, species, preprocessors=None):
    self.preprocessors = preprocessors
    self.species = species
    self.dataset = dataset

    if self.preprocessors is None:
      self.preprocessors = []
            
  def load(self, imagePaths, verbose=-1):
    print("loading")
    mypath = '/data/' + self.dataset + '/download/images'

    found = []
    for root, subFolders, files in os.walk(mypath):
      for folder in subFolders:
        for f in listdir(mypath + "/" + folder):
          found.append(mypath + "/" + folder + "/" + f)        

    data = []
    labels = []
    distinct_labels = []

    for (i, imagePath) in enumerate(imagePaths):
      if (len(distinct_labels) >= self.species):
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
