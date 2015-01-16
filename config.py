__author__ = 'Pierre Delaunay'


import glob
import numpy as np
# import copy
# import pandas as pd

# configuration file
# Define where the images are

image_folder = '/class/'
image_type = '.jpg'

max_col_size_bnw = 785664   # black and white
max_col_size_rgb = 2356992  # rgb

number_image = 25000

# getting
image_all = glob.glob(image_folder + 'train/' + '*' + image_type)
image_dog = glob.glob(image_folder + 'train/' + 'dog.' + '*' + image_type)
image_cat = glob.glob(image_folder + 'train/' + 'cat.' + '*' + image_type)

labels = np.load(image_folder + "labels.npy")


def to_row(a):
    return a.reshape((1, a.size))

# import pandas as pd
#mage_test = glob.glob(image_folder + 'test/' + '*' + image_type)
# image_dog = glob.glob(image_folder + 'train/' + 'dog.' + '*' + image_type)
# image_cat = glob.glob(image_folder + 'train/' + 'cat.' + '*' + image_type)


# extract labels
# all
# map_dog = dict()
# label_map = np.zeros((25000, 2))
# for i in image_all:
#     map_dog[i] = 0
#
# for i in image_dog:
#     map_dog[i] = 1
#
# j = 0
# for i in image_all:
#     if map_dog[i] == 1:
#         label_map[j, 0] = 1
#     j += 1
#
# label_map[:, 1] = 1 - label_map[:, 0]
#
# np.save(image_folder + "labels", label_map)
