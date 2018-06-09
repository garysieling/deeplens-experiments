mypath = "../training/frames"

import os
from os import listdir
from scipy.ndimage import imread 
import argparse
import logging
import random
import imutils
import json

import mxnet as mx
import numpy as np
from autoencoder import AutoEncoderModel

imagePaths = []
added = None
for root, subFolders, files in os.walk(mypath):
  for f in files:
    if (f.endswith("jpg")):
      imagePaths.append(root + "/" + f)

random.shuffle(imagePaths)

import cv2

WIDTH = 576
HEIGHT = 324
size = WIDTH * HEIGHT

X = [
  # (1080, 1920, 3)
  cv2.cvtColor(
    imutils.resize(
      imread(path),
      width = WIDTH
    ),
    cv2.COLOR_BGR2GRAY
  ).reshape(
    size
  )
  for path in imagePaths[0:1000]
]

hyperparameters={}
hyperparameters['X_SIZE'] = len(X)
hyperparameters['WIDTH'] = WIDTH

logging.basicConfig(level=logging.INFO)
print_every = 100
batch_size = 100
pretrain_num_iter = 50000
finetune_num_iter = 10000
visualize = False
gpu = False
eval_batch_size = 100

layers = str(size) + ",500,500,2000,10"
layers = [int(i) for i in layers.split(',')]

xpu = mx.gpu() if gpu else mx.cpu()
print("Training on {}".format("GPU" if gpu else "CPU"))

if (gpu):
  hyperparameters['device'] = 'GPU'
else:
  hyperparameters['device'] = 'CPU'

ae_model = AutoEncoderModel(xpu, layers, eval_batch_size, hyperparameters=hyperparameters, pt_dropout=0.2, internal_act='relu',
                                output_act='relu')

split_point = int(len(X)*0.75)
train_X = X[:split_point]
val_X = X[split_point:]


ae_model.layerwise_pretrain(train_X, batch_size, pretrain_num_iter, 'sgd', l_rate=0.001,
                              decay=0.0, lr_scheduler=mx.lr_scheduler.FactorScheduler(20000, 0.7),
                              print_every=print_every)

ae_model.finetune(train_X, batch_size, finetune_num_iter, 'sgd', l_rate=0.1, decay=0.0,
                  lr_scheduler=mx.lr_scheduler.FactorScheduler(20000, 0.1), print_every=print_every)

ae_model.save('autoencoder.arg')
ae_model.load('autoencoder.arg')

print("Training error:", ae_model.eval(train_X))
print("Validation error:", ae_model.eval(val_X))
if visualize:
  try:
    from matplotlib import pyplot as plt
    from model import extract_feature

    # sample a random image
    #index = np.random.choice(len(X))
    index = 0
    original_image = X[index]
    #print(json.dumps(original_image))
    data_iter = mx.io.NDArrayIter(
      {'data': [original_image]}, 
      batch_size=1, shuffle=False,
      last_batch_handle='pad')
    # reconstruct the image
    #  X_i = list(model.extract_feature(
    #                self.internals[i-1], self.args, self.auxs, data_iter, len(X),
    #                self.xpu).values())[0]

    reconstructed_image = extract_feature(ae_model.decoder, ae_model.args,
                                          ae_model.auxs, data_iter, 1,
                                          ae_model.xpu).values()[0]
    print("original image")
    plt.imshow(original_image.reshape((WIDTH, HEIGHT)))
    plt.show()
    print("reconstructed image")
    plt.imshow(reconstructed_image.reshape((WIDTH, HEIGHT)))
    plt.show()
  except ImportError:
    logging.info("matplotlib is required for visualization")