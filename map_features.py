# Used as reference:
# https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from imutils import paths
from matplotlib import pyplot as plt
import numpy as np
import pickle
import random
import os

MAP_FEATURE_PATH = 'input_data/chf/400.png'

model = load_model('cnn.model')         # Errors stemming from "no decode" should be solvable by downgrading h5py

map_path = MAP_FEATURE_PATH
file_name = os.path.split(map_path)[-1]
print(f'Feature map flag enabled: Plotting feature maps for {file_name}')

successive_outputs = [layer.output for layer in model.layers[1:]]

visualization_model = Model(inputs=model.input, outputs=successive_outputs)

img = load_img(map_path, target_size=(224, 224))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x = x / 255.0

successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]
print("Printing feature map shapes")
fig, axs = plt.subplots(5)
ax_index = 0
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)
    # Plot only the feature maps for non-FC layers
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]  # number of features in feature map
        size = feature_map.shape[1]  # feature map shape is (1, size, size, n_features)

        display_grid = np.zeros((size, size * n_features))

        for i in range(n_features):
            # Normalize features
            x = feature_map[0, :, :, i]
            x = x - x.mean()
            x = x / x.std()

            x = x * 64
            x = x + 128
            x = np.clip(x, 0, 255).astype('uint8')

            display_grid[:, i * size: (i + 1) * size] = x

        scale = 20. / n_features

        # plt.figure(figsize=(scale * n_features, scale))
        axs[ax_index].set_title(layer_name)
        plt.grid(False)
        axs[ax_index].imshow(display_grid, aspect='auto', cmap='viridis')
        ax_index += 1

plt.savefig(f'featuremap.png', format='png')

