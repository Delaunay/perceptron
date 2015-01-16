__author__ = 'midgard'

# trash file
# used to test python code snippet
# this file can be delete with no effect or what so ever on the main prog

import config as c
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


x = pd.read_csv('c:/class/test_data/x.csv', header=None).values
y = pd.read_csv('c:/class/test_data/y.csv', header=None).values
t1 = pd.read_csv('c:/class/test_data/t1.csv', header=None).values
t2 = pd.read_csv('c:/class/test_data/t2.csv', header=None).values



# select some image
# n = 50
#
# selected_label = []
# selected_image = []
#
# for i in range(0, n):
#     selected_label.append(c.labels[i, 0])
#     selected_label.append(c.labels[c.number_image - i - 1, 0])
#
#     selected_image.append(c.image_all[i])
#     selected_image.append(c.image_all[c.number_image - i - 1])
#
# data = pd.DataFrame({'img': selected_image, 'label': selected_label})
#
# b = np.zeros((n * 2, c.max_col_size_bnw), np.uint8)
# a = []
#
# for i in range(0, n * 2):
#     temp = c.to_row(np.array(Image.open(selected_image[i]).convert('L')))[0][:]
#     b[i, 0:temp.shape[0]] = temp
#
# import logistic as l
#
# theta = np.zeros((1, c.max_col_size_bnw))
# lr = l.LogisticRegression(b, np.array(selected_label))
#
#
# h = lr.solve_bfgs(theta)
# np.save(c.image_folder + "sol", h)
# print(h)


    # print(temp)
    # a =
    # pass



# im_array_dog = np.array(Image.open(image_dog[0]).convert('L'))

#
# def to_row(a):
#     return a.reshape((1, a.size))
#
#
# image_folder = '/class/'
# image_type = '.jpg'
#
# image_path = glob.glob(image_folder + 'train/' + '*' + image_type)
#
# max_size = 0
# for i in image_path:
#     max_size = max(np.array(Image.open(i)).size, max_size)
#
#
# print(max_size)



#
# image_dog = glob.glob(image_folder + 'train/' + 'dog.' + '*' + image_type)
# image_cat = glob.glob(image_folder + 'train/' + 'cat.' + '*' + image_type)
#
# # print(image_path)
# print(image_cat)
# print(image_dog)
#
# # black and white One point 1-bit pixels
# # matplotlib get the color wrong
# # im_array = np.array(Image.open(image_path[0]).convert('1'))
#
# # black and white 8-bit pixels
# # matplotlib get the color wrong..
# im_array_dog = np.array(Image.open(image_dog[0]).convert('L'))
# im_array_cat = np.array(Image.open(image_cat[0]).convert('L'))
#
# a = Image.fromarray(im_array_dog)
# a.save("test_dog.jpg", 'JPEG')
#
# a = Image.fromarray(im_array_cat)
# a.save("test_cat.jpg", 'JPEG')


# print(to_row(im_array))
#plt.imshow(im_array)
#plt.show()


#
# theta = np.array([[10, 10, 10, 10, 10, 10]])
#
# x = np.array([[10, 10, 10, 10, 10, 10],
#               [10, 10, 10, 10, 10, 10]])
#
#
# print(x)
# print(theta)
# print(logistic(x, theta))
#
# print("Mat  :" + str(np.zeros((2, 2))))     # Matrix
# print("Mat  :" + str(np.zeros((2, 1))))     # Matrix
# print("Mat  :" + str(np.zeros((1, 2))))     # Matrix
# print("Vec  :" + str(np.zeros(2)))          # Vector
#
# # How to cast a matrix into a vector
#
# print("Cast :" + str(np.zeros((1, 2))[0]))