# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import numpy as np
import copy

structure = {
    "input": 401,
    "hidden": [25],
    "output": 10,
}


Option = {
    "stochastic": True,
    "size": 100,

}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / x.sum()


def add_bias(x):
    xp = np.ones((x.shape[0], x.shape[1] + 1))
    xp[:, 1:] = x
    return xp


# def softmax_grad(x):
#     return softmax(x)


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def format_labels(y):
    """ Takes a vector Y containing multiple labels and return
    one matrix containing one labels per column handles class for [0:n] or to [1:n]"""

    n = np.min(y)
    yy = np.zeros((y.shape[0], np.max(y) - n + 1), dtype=np.float32)

    for i, label in enumerate(y):
        yy[i, label - n] = 1

    return yy


class Perceptron:

    def __init__(self, y, x, struct):

        # structure
        self.structure = struct
        self.size = self._generate_size()       # Matrix Size So people can see the NN structure

        self.y = y      # labels
        self.x = x      # data

        # Should I handle the case where x cols =/= input ?
        # Current - No: user may have forgotten bias unit
        # self.structure['input'] = x.shape[1]

        # save result of each layer || needed for the gradient
        self.a = []     # save feature matrix of each layer [1 X]
        self.z = []     # a * params
        self.g = []     # grad

        # save the last thing called prediction
        self.h = np.zeros(0)

        # generate params and size and allocate memory
        self.params = self._generate_params()   # initialize
        self.g = copy.deepcopy(self.params)     # grad same size params
        self._allocate_memory()                 # allocate h, z, and a

    def cost_function(self, set=None, l=0):
        """ if set is specified use set"""

        if set is None:
            x = self.x
            y = self.y
        else:
            x = set[0]
            y = set[1]

        theta = self.params
        end = self.end() - 1
        reg = 0

        # first layer
        self.a[0][:, 1:] = x
        self.z[0] = np.dot(self.a[0], np.transpose(theta[0]))

        for i in range(0, self.hidden_size() - 1):
            # self.a[i]
            print("here")
            pass

        self.a[end][:, 1:] = sigmoid(self.z[end - 1])
        self.z[end] = np.dot(self.a[end], np.transpose(theta[end]))

        self.h = sigmoid(self.z[end])

        # compute reg
        if l != 0:
            i = 0
            for el in self.params:
                reg += np.sum(el[:, 2:self.size[i][1]] ** 2)
                i += 1

            reg = reg * l / (2.0 * self.train_size())

        j = - (np.sum(np.log(self.h) * y) + np.sum(np.log(1.0 - self.h) * (1.0 - y))) / self.train_size()

        return j + reg

    def gradient(self, set=None, l=0):

        if set is None:
            y = self.y
        else:
            y = set[1]

        theta = self.params
        end = self.end() - 1
        self.g[end] = self.h - y                                                        # (5000x10)

        for i in range(1, end + 1):
            self.g[end - i] = self.g[end - i + 1].dot(theta[end - i + 1])               # (5000x10 10x26)
            self.g[end - i] = self.g[end - i][:, 1:] * sigmoid_grad(self.z[end - i])    # (5000x25 .* 5000x25)

        for i in range(0, end + 1):
            self.g[i] = np.transpose(self.g[i]).dot(self.a[i])                          # (5000x25 * 5000x401)

        # regularization
        if l != 0:
            for i in range(0, end):
                # only copy the first Columns
                # might be better than trying to 'delete' first column
                temp = np.zeros_like(self.params[i])
                temp[:, 0] = self.params[i][:, 0]
                self.g[i] += l * (self.params[i] - temp)

        return self.g

    def predict(self, x):

        t = x
        for i in self.params:
            t = sigmoid(add_bias(t).dot(np.transpose(i)))
        return t

    def accuracy(self, set):
        return np.mean(np.array_equal(self.predict(set[0]), set[1]))

    def stochastic_gradient(self, set, alpha, batch):

        x = set[0]
        y = set[1]

        sb = x.shape[0] / batch

        for i in range(0, sb):
            h = i * batch
            k = (i + 1) * batch

            # print(x)
            # print(x[h:k].shape)
            # print(y[h:k].shape)

            self.gradient((x[h:k, :], y[h:k]))

            for j in range(0, len(self.g)):
                self.params[j] -= alpha * self.g[j]

            print(self.cost_function((x[h:k, :], y[h:k])))

    def gradient_descent(self, alpha, max_ite):

        # print(len(self.params))
        # print(len(self.g))
        old = self.cost_function()

        for i in range(0, max_ite):
            for j in range(0, len(self.g)):
                self.params[j] -= alpha * self.g[j]

            new = self.cost_function()
            print(new, new-old)
            old = new

    def numerical_gradient(self):
        pass

    def end(self):
        return self.hidden_size() + 1

    def output(self):
        return self.structure['output']

    def input(self):
        return self.structure['input']

    def hidden(self, k):
        return self.structure['hidden'][k]

    def hidden_size(self):
        return len(self.structure['hidden'])

    def last_hidden(self):
        return self.hidden(self.hidden_size() - 1)

    def train_size(self):
        return self.x.shape[0]

    def theta(self, k):
        return self.params[k]

    def _generate_size(self):
        """ Generate the size of each set of params as a numpy shape tuple"""

        size = list()
        # Input Layer
        size.append((self.hidden(0), self.input() + 1))

        # hidden layer
        for i in range(0, self.hidden_size() - 1):
            size.append((self.hidden(i + 1), self.hidden(i) + 1))

        # output layer
        size.append((self.output(), self.last_hidden() + 1))

        return size

    def _allocate_memory(self):

        self.a.append(np.ones((self.train_size(), self.input() + 1)))  # 5000x401
        self.z.append(np.ones((self.train_size(), self.hidden(0))))    # 5000x25

        for i in range(0, self.hidden_size() - 1):
            self.a.append(np.ones((self.train_size(), self.hidden(i) + 1)))    # Skipped
            self.z.append(np.ones((self.train_size(), self.hidden(i + 1))))

        self.a.append(np.ones((self.train_size(), self.last_hidden() + 1)))    # 5000x26
        self.z.append(np.ones((self.train_size(), self.output())))             # 5000x10

    def _generate_params(self):
        """allocate memory for params and initialize them with random numbers"""
        params = list()

        for i in self.size:
            params.append(np.random.uniform(0, 1, i))

        return params


if __name__ == "__main__":

    y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    y = format_labels(y)

    print(y)

    x = np.array([
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40],
        [1, 20, 30, 40]
    ])

    a = Perceptron(y, x, structure)

    print(a.size)


    # print(a.a[0].shape)
