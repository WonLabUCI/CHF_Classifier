from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2
import time

#python predict.py -i input_data/chf/0.png -m model.h5 -l lb.pickle -w 448 -ht 448

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image used for prediction')
ap.add_argument('-m', '--model', required=True, help='path to trained model')
ap.add_argument('-l', '--label-bin', required=True, help='path to label binarizer')
ap.add_argument('-w', '--width', required=True, help='input width of model')
ap.add_argument('-ht', '--height', required=True, help='input height of model')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
output = image.copy()
image = cv2.resize(image, (int(args['width']), int(args['height'])))

image = image.astype('float')/255.0

image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))

print('[INFO] Loading model and label binarizer')
model = load_model(args['model'])
lb = pickle.loads(open(args['label_bin'], 'rb').read())

timer_start = time.perf_counter()

preds = model.predict(image)

timer_end = time.perf_counter()

print(f'Time elapsed: {timer_end - timer_start:0.4f} seconds')

confidence = preds[0][0]
index = round(confidence)
print(f'Confidence: {confidence}')
print(f'Index: {index}')

# i = preds.argmax(axis=1)[0]
'''
print(f'preds: {preds}')
print(f'preds[0]: {preds[0]}')
print(f'preds[0][0]: {preds[0][0]}')
print(f'preds.argmax(): {preds.argmax(axis=1)}')
print(f'preds.argmax()[0]: {i}')
print(f'preds[preds.argmax(axis=1)[0]]: {preds[i]}')
print(f'i: {i}')
print(lb.classes_)
'''

# label = lb.classes_[i]

label = lb.classes_[index]

if index == 0:
    text = f'{label}: {(1-preds[0][0])*100:.2f}%'       # class is CHF, use confidence of CHF
else:
    text = f'{label}: {(preds[0][0])*100:.2f}%'   # class is nonCHF, use confidence of nonCHF

cv2.putText(output, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Image', output)
cv2.waitKey(0)
cv2.imwrite('output/prediction.png', output)

