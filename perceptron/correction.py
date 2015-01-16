# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'


import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    return x


def compute_h(x, w, b):
    return sigmoid(np.dot(x, w) + b)


def compute_y(h, v, c):
    y2 = np.exp(np.dot(h, v) + c)
    return y2 / y2.sum(axis=1, keepdims=True)


def compute_loss(y, t):
    return -(t * np.log(y)).sum(axis=1).mean()


def compute_missclass(y, t):
    return (np.not_equal(t.argmax(axis=1), y.argmax(axis=1))).mean()


def forward_propagation(x, t, w, b, v, c):

    h = compute_h(x, w, b)
    y = compute_y(h, v, c)
    loss = compute_loss(y, t)
    missclass = compute_missclass(y, t)

    return h, y, loss, missclass


def backward_propagation(x, t, h, y, v):

    # np.outer(h, y - t)
    # grad_v = (h[:, :, None] * (y - t)[:, None, :]).mean(axis=0)
    grad_v = np.dot(h, t, y - t) / h.shape[0]

    grad_d = (y - t).mean(axis=0)
    grad_w = (np.dot(y - t, v.t) * h * (1 - h)[:, None, :] * x[:, :, None]).mean(axis=0)
    grad_b = (np.dot(y - t, v.t) * h * (1 - h)).mean(axis=0)

    return grad_v, grad_d, grad_w, grad_b


def check_gradient(x, t, w, b, v, c, delta=1e-5):
    h = compute_h(x, w, b)
    y = compute_y(h, v, c)
    loss = compute_loss(y, t)
    grad_v, grad_d, grad_w, grad_b = backward_propagation(x, t, h, y, v)

    b2 = np.copy(b)
    b_grad = np.zeros_like(b)

    for i in xrange(b.shape[0]):
        b2[i] = b[i] + delta
        h2 = compute_h(x, w, b2)
        y2 = compute_y(h2, v, c)
        loss2 = compute_loss(y2, t)

        b_grad[i] = (loss2 - loss)/delta
        b2[i] = b[i]

    np.testing.assert_allclose(b_grad, grad_b, atol=1e-5)