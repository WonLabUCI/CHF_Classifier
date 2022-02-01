from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import data_loading

data = []
labels = []
data_loading.load_bubble_images(data, labels, 'input_data')

print('LOADING IMAGES')
data = np.array(data, dtype='float')/255.0
labels = np.array(labels)

print('PARTITIONING DATA')
(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.25, random_state=22)
image_shape = trainX[0].shape






