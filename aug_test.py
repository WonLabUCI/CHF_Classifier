
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

aug = ImageDataGenerator(horizontal_flip=True, brightness_range=[0.3,1.7], fill_mode='nearest')

img = load_img('input_data/chf/0.png')
img_array = img_to_array(img)
img_array = img_array/255.0
img_array = img_array.reshape((1,)+img_array.shape)

count = 0
for img in aug.flow(img_array, batch_size=1,save_to_dir='augs',save_prefix='aug', seed=100):
    count += 1
    if count > 25:
        break
print('COMPLETE')