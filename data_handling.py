# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import numpy as np
import scipy.linalg as blas

#   Build basic data handling
#       - load data set into memory (separate training set into training/validation and keep the test set)
#       - statistic about the data set (max feature size)
#       - basic data transform
#           - PCA to reduce the program footprint
#           - Color/Black & White
#       - Should I save transformed data ?
#       - this class may also handle the case were we load image n by n to limit the program footprint
#       (stochastic gradient)

# 1 Row = 1 Sample
# 1 Col = 1 Feature

dog = [0, 1]
cat = [1, 0]

# option JSON style
option_gen = {

    # when loading data in batch for stochastic Gradient Descent
    "batch": {
        "active": False,
        "size": 1000
    },

    # disabled when using stochastic Gradient Descent
    "PCA": {
        "active": False,
        "dim": 4
    },

    # when using scaling
    "scaling": False,

    # color vs Black and white
    "color": True,
}


class DataHandling():

    option = option_gen

    pca_result = {
        "U": None,
        "S": None
    }

    scaling_result = {
        "mean": None,
        "std": None
    }

    # used to find feature parameters
    training = {
        "features": None,
        "labels": None
    }

    # used to select models parameters (reg fact, early stopping ...
    validation = {
        "features": None,
        "labels": None
    }

    test = {
        "features": None,
        "labels": None
    }

    def __init__(self, option=dict()):

        # replace defaults by provided option
        for i in option:
            self.option[i] = option[i]

    def transform_element_wise(self):
        pass

    def transform_whole(self):
        pass

    @staticmethod
    def handle_size():
        """ handle different size of photos """
        pass

    @staticmethod
    def handle_color():
        """ return a black and white version of the photo"""
        pass

    def pca(self, x, dim):
        """ reduce the size of the feature matrix to dim"""

        if self.u() is None:

            x = self.scaling(x)

            cov = x.transpose().dot(x) / float(len(x))
            val = blas.svd(cov)

            self.pca_result["U"] = val[0]
            self.pca_result["S"] = val[1]

        return x.dot(self.u()[:, 0:dim])

    def pca_retained(self, dim):
        """ Compute retained variance"""
        if self.s() is None:
            raise ZeroDivisionError("You Should call pca() first... Have you ?")

        return self.s()[0:dim].sum() / self.s().sum()

    def pca_reconstruct(self, z):
        """ reconstruct the compressed data ( * std + mean)"""

        if self.u() is None:
            raise ZeroDivisionError("You Should call pca() first... Have you ?")

        dim = z.shape[1]
        return z.dot(self.u().transpose()[0:dim, :]) * self.std() + self.mean()

    def scaling(self, x):
        """ normalize the data"""

        if self.mean() is None:
            self.scaling_result["mean"] = x.mean(axis=0)
            self.scaling_result["std"] = x.std(axis=0)

        return (x - self.mean()) / self.std()

    # Shortcuts
    def batch(self):
        return self.option["batch"]["active"]

    def batch_size(self):
        return self.option["batch"]["size"]

    def pca_transform(self):
        return self.option["PCA"]["active"]

    def pca_dim(self):
        return self.option["PCA"]["dim"]

    def has_scaling(self):
        return self.option["scaling"]

    def has_color(self):
        return self.option["color"]

    def is_black_white(self):
        return not self.has_color()

    def u(self):
        return self.pca_result["U"]

    def s(self):
        return self.pca_result["S"]

    def mean(self):
        return self.scaling_result["mean"]

    def std(self):
        return self.scaling_result["std"]