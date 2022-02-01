"""## Preparing Data"""

from matplotlib import pyplot as plt

from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix

import data_loading
from imutils import paths
import argparse
from natsort import os_sorted


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset of images')
#ap.add_argument("-m", "--model", required=True, help="path to output trained model")
#ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# data_dir = "/content/drive/MyDrive/CNNLSTM/video_data"
data_dir = args['dataset']
img_height, img_width = 64, 64
seq_len = 70

classes = ["chf", "nonchf"]


def create_data(input_dir):
    X = []
    Y = []

    classes_list = os.listdir(input_dir)

    for c in classes_list:
        print(c)
        #files_list = os_sorted(os.listdir(os.path.join(input_dir, c)))
        files_list = os_sorted(paths.list_images(os.path.join(input_dir, c)))
        #print(files_list)
        frames = []

        # Adapting loop for image data instead of video
        for f in files_list:
            #print(f)

            image = cv2.imread(f)
            image = cv2.resize(image, (img_height, img_width))

            frames.append(image)

            if len(frames) == seq_len:
                X.append(frames)
                print('len(X):', len(X))

                y = [0] * len(classes)
                y[classes.index(c)] = 1
                Y.append(y)

                frames = []

        '''
        for f in files_list:
            # print(f)
            frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
            if len(frames) == seq_len:
                X.append(frames)
                # print('len(X):', len(X))

                y = [0] * len(classes)
                y[classes.index(c)] = 1
                Y.append(y)
        '''

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


"""## Create Data"""

print('[INFO] Loading images')

X, Y = create_data(data_dir)
print('X.shape:', X.shape)
print('Y:', Y)

"""## Split into Train and Test Datasets"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)
print('X_train:', X_train.shape)
print('y_test:', y_test.shape)




"""## ConvLSTM Based Model Design"""

model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False, data_format="channels_last",
                     input_shape=(seq_len, img_height, img_width, 3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))

"""## Model Summary"""

model.summary()

"""## Training the Model"""

opt = optimizers.SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

earlystop = EarlyStopping(patience=7)
callbacks = [earlystop]

num_epochs = 40

# Reduced batch_size from 8 to 4 to reduce memory costs
history = model.fit(x=X_train, y=y_train, epochs=num_epochs, batch_size=5, shuffle=True, validation_split=0.2,
                    callbacks=callbacks)

"""## Evaluate Model"""

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))


'''
print('[INFO] Plotting training statistics')
N = np.arange(0, num_epochs)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, history.history['loss'], label='train_loss')
plt.plot(N, history.history['val_loss'], label='val_loss')
plt.plot(N, history.history['accuracy'], label='train_acc')
plt.plot(N, history.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['plot'])
'''


