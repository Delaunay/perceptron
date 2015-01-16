__author__ = 'Pierre Delaunay'

import numpy as np
import scipy.optimize as solver
import matplotlib.pyplot as plt


def sigmoid(x):
        return 1.0 / (1.0 + np.exp(- x))


class LogisticRegression():
    """ Logistic regression :: for one class only"""

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.m = self.y.shape[0]

    def cost(self, theta):
        x = theta.dot(np.transpose(self.x))
        return (np.sum(np.log(1.0 + np.exp(- x))) + x.dot(1 - self.y))/self.m

    # def gradient(self, theta):
    #
    #     # makes sure the numpy array is (1, n) and not (n, ) ...
    #     # and yes you need to cast it first into a matrix =_=
    #     if theta.shape[0] == 1:
    #         theta = np.array(np.matrix(theta))
    #
    #     h = sigmoid(self.x.dot(np.transpose(theta)))
    #
    #     return np.sum(np.transpose(h - self.y).dot(self.x), axis=0) / self.m
    #
    # def solve_gradient_descent(self, theta, ite, al=0.10):
    #     # doesnot work because the gradient is wrong because numpy sucks
    #     for i in range(0, ite):
    #         theta -= al * self.gradient(theta)
    #
    #     return theta
    #
    # def gd_debug(self, theta, ite, al=0.10):
    #
    #     hist = []
    #
    #     for i in range(0, ite):
    #         theta -= al * self.gradient(theta)
    #         hist.append(self.cost(theta))
    #
    #     print(hist)
    #     plt.plot(hist)
    #     plt.show()

    def solve_bfgs(self, theta):
        return solver.fmin_l_bfgs_b(self.cost, theta)


if __name__ == '__main__':

    # load data
    import pandas as pd

    xx = pd.read_csv("X", ' ', header=None).values[:, 1:]
    yy = pd.read_csv("y", ' ', header=None).values[:, 1:]

    thetax = np.zeros((1, 3))
    thetay = np.array([[-25.16, 0.2062, 0.2014]])

    lr = LogisticRegression(xx, yy)

    print("Cost")
    print(lr.cost(thetax))
    print(lr.cost(thetay))

    print("Gradient")
    print(lr.gradient(thetax))
    print(lr.gradient(thetay))

    t = lr.solve_bfgs(thetax)
    print(t)
    print(lr.gradient(t))





