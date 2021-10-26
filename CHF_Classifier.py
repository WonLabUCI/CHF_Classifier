# Referenced heavily:
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/

# imutils package was downloaded and added to root directory despite also being pip-installable
# Since this package seems to be fairly uncommon (written and published by Adrian Rosebrock),
# I decided to include it in the root directory to avoid adding another dependency

# RUN COMMAND: python CHF_Classifier.py -d input_data -m output\cnn.model -l output\lb.pickle -p output\plot.png

# Import packages
import matplotlib
matplotlib.use('Agg')

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

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset of images')
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

print('[INFO] Loading images')
data = []
labels = []

# Each "image" is encoded in data as a (640, 106, 3) numpy array
# (height, width, channels)
data_loading.load_bubble_images(data, labels, args["dataset"])

print('[INFO] Images successfully loaded')

# Scale raw pixel intensities to a range of [0,1]
data = np.array(data,dtype='float')/255.0
labels = np.array(labels)

# Partition data into training and test sets
# random_state parameter is effectively a seed, using a predetermined one allows for reproducibility
# X is image data, Y is label data
print('[INFO] Partitioning data')
(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.25, random_state=22)
image_shape = trainX[0].shape

# In order to translate the labels into integer labels and perform an appropriate one-hot encoding, strings are
# encoded using LabelEncoder, then the encoded vector is used to generate an actual one-hot encoding
# using to_categorical()
print('[INFO] Performing encoding on labels')

print('[INFO] Encoding is being done with LabelBinarizer')
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print(f'[TEST] Printing LabelBinarizer classes: {lb.classes_}')
print(f'[TEST] Printing label associated with 0: {lb.inverse_transform(np.array([0]))}')
#print(lb.classes_)
#print(testY)


'''
le = LabelEncoder()
le.fit(testY)
testY = le.transform(testY)
trainY = le.transform(trainY)

#print(testY_encoding)

testY = to_categorical(testY, num_classes=2)
trainY = to_categorical(trainY, num_classes=2)
#print('[TEST] Printing encoded labels')
#print(test_categorical)
#print(le.classes_)
'''


aug = ImageDataGenerator(horizontal_flip=True, fill_mode='nearest')

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=image_shape))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding='same', input_shape=image_shape))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same', input_shape=image_shape))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

print('[INFO] Training network')

# Set parameters for training
LR = 1e-4
batch_size = 32
num_epochs = 75
opt_decay = LR/num_epochs

opt = Adam(lr=LR, decay=opt_decay)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

H = model.fit(x=aug.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX,testY),
                steps_per_epoch=len(trainX)//batch_size, epochs=num_epochs)

print('[INFO] Evaluating network')
predictions = model.predict(x=testX, batch_size=32)
print(predictions)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=lb.classes_, labels=[0,1]))

# Plot training loss and accuracy
print('[INFO] Plotting training statistics')
N = np.arange(0, num_epochs)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.plot(N, H.history['accuracy'], label='train_acc')
plt.plot(N, H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['plot'])

# Save model and label binarizer
print('[INFO] Serializing network and label binarizer')
model.save(args['model'], save_format='h5')
f = open(args['label_bin'], 'wb')
f.write(pickle.dumps(lb))
f.close()









