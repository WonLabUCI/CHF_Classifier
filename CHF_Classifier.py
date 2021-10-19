# Referenced heavily:
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/

# imutils package was downloaded and added to root directory despite also being pip-installable
# Since this package seems to be fairly uncommon (written and published by Adrian Rosebrock),
# I decided to include it in the root directory to avoid adding another dependency

# Import packages
import matplotlib
matplotlib.use('Agg')


from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset of images')
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binarizer")

# Consider making the plot argument not required, then only plot if the args map contains a -p entry
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

print('[INFO] Loading images')
data = []
labels = []

data_loading.load_bubble_images(data, labels, args["dataset"])

print('[INFO] Images successfully loaded')

#print(data)
#print(labels)

# Scale raw pixel intensities to a range of [0,1]
data = np.array(data,dtype='float')/255.0
labels = np.array(labels)

# Partition data into training and test sets
# random_state parameter is effectively a seed, using a predetermined one allows for reproducibility
# X is image data, Y is label data
print('[INFO] Partitioning data')
(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.25, random_state=22)

print('[TEST] Printing test labels')
print(testY)
print('[TEST] Printing number of test labels')
print(len(testY))

#trainY = to_categorical(trainY, num_classes=2)
#testY = to_categorical(testY, num_classes=2)

#lb = LabelBinarizer()
#trainY = lb.fit_transform(trainY)
#testY = lb.transform(testY)

#OH_encoder = OneHotEncoder()
#trainY = OH_encoder.fit_transform(np.array(trainY))
#testY = OH_encoder.transform(np.array(testY))

# In order to translate the labels into integer labels and perform an appropriate one-hot encoding, strings are
# encoded using LabelEncoder, then the encoded vector is used to generate an actual one-hot encoding
# using to_categorical()
print('[INFO] Performing one-hot encoding on labels')
le = LabelEncoder()
le.fit(testY)
testY_encoding = le.transform(testY)

#print(testY_encoding)

test_categorical = to_categorical(testY_encoding, num_classes=2)
print('[TEST] Printing encoded labels')
print(test_categorical)

aug = ImageDataGenerator(horizontal_flip=True, fill_mode='nearest')

#print(testY)
#print(len(testY))






