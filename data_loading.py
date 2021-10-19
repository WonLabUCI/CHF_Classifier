from imutils import paths
import cv2
import os



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

    #print('LOAD_BUBBLE_IMAGES CALLED')

    # Retrieve image paths
    image_paths = sorted(list(paths.list_images(datapath)))

    # pyimagesearch data loading suggests shuffling data, not sure how important this is. Look into it

    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Pretty sure these dimensions are kind of HUGE, even if it's going to be downscaled
        # Since these images have not been flattened (I believe we should avoid flattening for CNNs?),
        # each image is effectively 631x3840x3=7,269,120 pixels
        # For reference, the images used in the DeepBoiling model were 224x224x3=150,528 pixels

        # Diving each parameter by 6 to maintain a relatively similar aspect ratio
        image = cv2.resize(image, (106, 640))
        data.append(image)

        # Since the directory format will likely have separate folders for CHF and nonCHF classes, we should probably
        # source the label information from the path itself

        # -2 index will take the name of the directory directly above the image
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)





