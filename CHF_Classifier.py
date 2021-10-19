# Referenced heavily:
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/

# imutils package was downloaded and added to root directory despite also being pip-installable
# Since this package seems to be fairly uncommon (written and published by Adrian Rosebrock),
# I decided to include it in the root directory to avoid adding another dependency

# Import packages
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import data_loading

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', require=True, help='path to input dataset of images')
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binarizer")

# Consider making the plot argument not required, then only plot if the args map contains a -p entry
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

print('[INFO] Loading images')
data = []
labels = []

data_loading.load_bubble_images(data, labels, args["dataset"])





