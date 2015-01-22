__author__ = 'Pierre Delaunay'


import glob

# configuration file
# Define where the image are

image_folder = '/class/'
image_type = '.jpg'

max_col_size_bnw = 785664   # black and white
max_col_size_rgb = 2356992  # rgb

number_image = 25000

# getting
image_all = glob.glob(image_folder + 'train/' + '*' + image_type)


def to_row(a):
    return a.reshape((1, a.size))


# import pandas as pd
#mage_test = glob.glob(image_folder + 'test/' + '*' + image_type)
# image_dog = glob.glob(image_folder + 'train/' + 'dog.' + '*' + image_type)
# image_cat = glob.glob(image_folder + 'train/' + 'cat.' + '*' + image_type)