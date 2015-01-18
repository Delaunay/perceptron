# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import numpy as np


def shuffle(x, y, yl):

    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    byl = np.zeros_like(yl)
    ck = {}
    idx = 0

    for i in range(len(x)):

        while True:

            idx = np.random.uniform(0, len(x))

            if idx not in ck:
                ck[idx] = 1
                break

        ck[idx] = 1

        byl[i, :] = yl[idx, :]
        bx[i, :] = x[idx, :]
        by[i, :] = y[idx, :]

    return bx, by, byl


import scipy.linalg as blas


def line_search(f):
    """ I am not sure if line search is the term but..."""
    # 1 Quadratic approximation | interpolation to find an estimate of the
    # value x which minimize f(t - x * grad)
    # better thn the linear approximation

    a = np.array([
        [0.0,        0.0, 1.0],
        [0.5 ** 2.0, 0.5, 1.0],
        [1.0,        1.0, 1.0],
    ])

    b = np.array([
        [f(0.0)],
        [f(0.5)],
        [f(1.0)]
    ])

    x = blas.solve(a, b)

    return - x[1, 0] / (2.0 * x[0, 0])

    # # 2 Linear Approximation:
    # x1 = 0.0
    # x2 = 0.25
    # fx1 = f(x1)
    # fx2 = f(x2)
    #
    # a1 = (fx1 - fx2)/(x1 - x2)
    # b1 = fx1 - a1 * x1
    #
    # x11 = 1.0
    # x21 = 0.75
    # fx11 = f(x11)
    # fx21 = f(x21)
    #
    # a2 = (fx11 - fx21)/(x11 - x21)
    # b2 = fx11 - a2 * x11
    #
    # return - (b1 - b2) / (a1 - a2)


def format_labels(y):
    """ Takes a vector Y containing multiple labels and return
    one matrix containing one labels per column handles class for [0:n] or to [1:n]"""

    n = np.min(y)
    yy = np.zeros((y.shape[0], np.max(y) - n + 1), dtype=np.float32)

    for i, label in enumerate(y):
        yy[i, label - n] = 1

    return yy


def hns(n):
    """ HNS function : Handle Numpy nonsense"""

    if len(n.shape) != 1:
        if n.shape[0] != 1:
            n = np.transpose(n)

        return n[0]

    return n


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    y = np.exp(x)
    return y / y.sum()


def add_bias(x):
    xp = np.ones((x.shape[0], x.shape[1] + 1))
    xp[:, 1:] = x
    return xp


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)