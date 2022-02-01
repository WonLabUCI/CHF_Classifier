from imutils import paths
import cv2
import os
import random



"""
load_bubble_images Design considerations:
Image sizing will be dependent on the number of frames in each image--could consider using a parameter for frames?

Parameters:
data should be an empty list
labels should be an empty list
datapath is a string describing the path to the dataset
Modifying contents of list arguments will modify them outside of the function's due to python's parameter-passing style
"""


def load_bubble_images(data, labels, datapath):

    # Retrieve image paths
    image_paths = sorted(list(paths.list_images(datapath)))
    random.seed(100)
    random.shuffle(image_paths)

    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Diving each parameter by 6 to maintain a relatively similar aspect ratio
        # CV2 USES SHAPE (Width, Height)
        #image = cv2.resize(image, (106, 640))
        image = cv2.resize(image, (224,224))
        #image = cv2.resize(image, (448, 448))
        data.append(image)

        # Since the directory format will likely have separate folders for CHF and nonCHF classes, we should probably
        # source the label information from the path itself

        # -2 index will take the name of the directory directly above the image
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)




if __name__ == '__main__':
    data = []
    labels = []
    load_bubble_images(data, labels, 'bubble_data')

    cv2.imwrite('test.png', data[0])



