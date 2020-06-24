# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
import scipy.misc
import scipy.ndimage
import glob


def read_data(path):
    with h5py.File(path,'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        data = np.transpose(data,[0,2,3,1])
        label = np.transpose(label, [0, 2, 3, 1])
        return data,label


def preprocess(path, scale=3):
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)
  label_ = label_ / 255.
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
  return input_, label_

def imread(path, is_grayscale=True):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def prepare_data( dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

    return data

def imsave(image, path):
  return scipy.misc.imsave(path, image)