__author__ = 'Pierre Delaunay'

# reads all the image and save them as a binary representation of a numpy array
# quicker to load

import config as c
from PIL import Image
import numpy as np

# # reduce the size of the set
# full_size = 10000
# train_size = int(0.6 * full_size)
# cross_size = int(0.2 * full_size)
# test = int(0.2 * full_size)
#
# c.image_all = c.image_all[0:full_size]
#
# # 15 000 the training example
# train = c.image_all[0:train_size]
#
# # 5000 the cross validation example
# cross_val = c.image_all[train_size:train_size + test]
#
# # 5000 the test example
# test = c.image_all[train_size + test:train_size + 2 * test]
#
# # create the full numpy array
# full = [np.array(Image.open(i).convert('L')) for i in test]
# np.save(c.image_folder + 'test_bnw', np.array(full))

print(np.load(c.image_folder + 'test_bnw.npy'))




