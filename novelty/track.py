mypath = "../training/frames"

import os
from os import listdir
from scipy.ndimage import imread 
import argparse
import logging
import random
import imutils

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

size = 20736

X = [
  # (1080, 1920, 3)
  cv2.cvtColor(
    imutils.resize(
      imread(path),
      width = 192
    ),
    cv2.COLOR_BGR2GRAY
  ).reshape(
    size
  )
  for path in imagePaths[0:100]
]

logging.basicConfig(level=logging.INFO)
print_every = 1000
batch_size = 20
pretrain_num_iter = 50000
finetune_num_iter = 100000
visualize = True
gpu = False

layers = [int(i) for i in "20736,500,500,2000,10".split(',')]

xpu = mx.gpu() if gpu else mx.cpu()
print("Training on {}".format("GPU" if gpu else "CPU"))

ae_model = AutoEncoderModel(xpu, layers, pt_dropout=0.2, internal_act='relu',
                                output_act='relu')

train_X = X[:50  ]
val_X = X[50:]


ae_model.layerwise_pretrain(train_X, batch_size, pretrain_num_iter, 'sgd', l_rate=0.1,
                              decay=0.0, lr_scheduler=mx.lr_scheduler.FactorScheduler(20000, 0.1),
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
    original_image = X[np.random.choice(X.shape[0]), :].reshape(1, size)
    data_iter = mx.io.NDArrayIter({'data': original_image}, batch_size=1, shuffle=False,
                                  last_batch_handle='pad')
    # reconstruct the image
    reconstructed_image = extract_feature(ae_model.decoder, ae_model.args,
                                          ae_model.auxs, data_iter, 1,
                                          ae_model.xpu).values()[0]
    print("original image")
    plt.imshow(original_image.reshape((28, 28)))
    plt.show()
    print("reconstructed image")
    plt.imshow(reconstructed_image.reshape((28, 28)))
    plt.show()
  except ImportError:
    logging.info("matplotlib is required for visualization")