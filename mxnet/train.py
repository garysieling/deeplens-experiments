from mxnet.gluon.model_zoo import vision

import os
import random
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet.test_utils import get_cifar10
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.contrib.io import DataLoaderIter

print(mx.__version__)

# N x 3 x H x W


def get_imagenet_transforms(data_shape=224, dtype='float32'):
  def train_transform(image, label):
    image, _ = mx.image.random_size_crop(image, (data_shape, data_shape), 0.08, (3/4., 4/3.))
    image = mx.nd.image.random_flip_left_right(image)
    image = mx.nd.image.to_tensor(image)
    image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return mx.nd.cast(image, dtype), label

  def val_transform(image, label):
    image = mx.image.resize_short(image, data_shape + 32)
    image, _ = mx.image.center_crop(image, (data_shape, data_shape))
    image = mx.nd.image.to_tensor(image)
    image = mx.nd.image.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return mx.nd.cast(image, dtype), label

  return train_transform, val_transform


def get_imagenet_iterator(root, batch_size, num_workers, data_shape=224, dtype='float32'):
  """Dataset loader with preprocessing."""
  train_dir = os.path.join(root, 'train')
  train_transform, val_transform = get_imagenet_transforms(data_shape, dtype)
  logging.info("Loading image folder %s, this may take a bit long...", train_dir)
  train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
  train_data = DataLoader(train_dataset, batch_size, shuffle=True,
                          last_batch='discard', num_workers=num_workers)
  
  val_dir = os.path.join(root, 'val')
  if not os.path.isdir(os.path.expanduser(os.path.join(root, 'val', 'n01440764'))):
    user_warning = 'Make sure validation images are stored in one subdir per category, a helper script is available at https://git.io/vNQv1'
    raise ValueError(user_warning)
  
  logging.info("Loading image folder %s, this may take a bit long...", val_dir)
  val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
  val_data = DataLoader(val_dataset, batch_size, last_batch='keep', num_workers=num_workers)
  return DataLoaderIter(train_data, dtype), DataLoaderIter(val_data, dtype)


resnet18 = vision.resnet18_v1(pretrained=True)
alexnet = vision.alexnet(pretrained=True)
inception = vision.inception_v3(pretrained=True)
#squeezenet = vision.squeezenet1_0()
#densenet = vision.densenet_161()

get_imagenet_iterator("c:\\data\\images", batch_size, num_workers, data_shape=224, dtype='float32'):