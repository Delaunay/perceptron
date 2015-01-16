# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import numpy as np
import copy

Options = {
    "stochastic": True,
    "size": 100,

    "structure": {
        "input": 400,
        "hidden": [25],
        "output": 10,
    }
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

    def __init__(self, y, x, option):

        # structure
        self.structure = option['structure']
        self.option = option
        self.size = self._generate_size()       # Matrix Size So people can see the NN structure
                                                # [0] : rows [1] : cols [2]: cumulative size (to reshape the col)

        self.y = y      # labels
        self.x = x      # data

        # Should I handle the case where x cols =/= input ?
        # Current - No: user may have forgotten bias unit
        # self.structure['input'] = x.shape[1]

        # save result of each layer || needed for the gradient
        self.a = []     # save feature matrix of each layer [1 X]
        self.z = []     # a * params

        # save the last thing called prediction
        self.h = np.zeros(0)

        # generate params and size and allocate memory
        self.params = self._generate_params()   # initialize
        self.g = list()                         # temporary
        self.grad = copy.deepcopy(self.params)  # grad
        self._allocate_memory()                 # allocate h, z, and a

        self.state = 0

    def theta(self, params, k):
        """ reshape params into the equivalent matrix"""

        if k == 0:
            return (params[0:self.size[0][2]]).reshape(self.size[0][0:2])
        else:
            return (params[self.size[k - 1][2]:self.size[k][2]]).reshape(self.size[k][0:2])

    def _forward_propagation(self, params):
        end = self.end() - 1

        self.a[0][:, 1:] = self.x
        self.z[0] = np.dot(self.a[0], np.transpose(self.theta(params, 0)))

        for i in range(0, self.hidden_size() - 1):

            self.a[i + 1][:, 1:] = sigmoid(self.z[i])
            self.z[i + 1] = np.dot(self.a[i + 1], np.transpose(self.theta(params, i + 1)))

        self.a[end][:, 1:] = sigmoid(self.z[end - 1])
        self.z[end] = np.dot(self.a[end], np.transpose(self.theta(params, end)))

        self.h = sigmoid(self.z[end])

        self.state = 1

    def cost_function(self, params, l=1.0):
        """ if set is specified use set"""

        # always compute the forward prop
        self._forward_propagation(params)
        reg = 0

        # compute reg
        if l != 0:
            for i in range(0, self.end()):
                reg += np.sum(self.theta(params, i)[:, 1:] ** 2)

            reg = reg * l / (2.0 * self.train_size())

        j = - (np.sum(np.log(self.h) * self.y) + np.sum(np.log(1.0 - self.h) * (1.0 - self.y))) / self.x.shape[0]

        return j + reg

    def gradient(self, params, l=1.0):

        # forward prop must be done to compute the gradient
        if self.state == 0:
            self._forward_propagation(params)

        end = self.end() - 1
        self.g[end] = self.h - self.y

        sig3 = self.h - self.y
        sig2 = sig3.dot(self.theta(params, 1))[:, 1:] * sigmoid_grad(self.z[0])

        if l < 0:
            self.grad[0:self.size[0][2]] = (np.transpose(sig2).dot(self.a[0])).reshape((self.size[0][2],))
            self.grad[self.size[0][2]:self.size[1][2]] = (np.transpose(sig3).dot(self.a[1])).reshape((self.size[1][0] * self.size[1][1],))
            self.state = 0
            self.grad /= self.x.shape[0]
            return self.grad

        else:
            temp = np.zeros_like(self.theta(params, 0))
            temp[:, 1:] = self.theta(params, 0)[:, 1:] * l
            self.grad[0:self.size[0][2]] = (np.transpose(sig2).dot(self.a[0]) + temp).reshape((self.size[0][2],))
            temp = np.zeros_like(self.theta(params, 1))
            temp[:, 1:] = self.theta(params, 1)[:, 1:] * l
            self.grad[self.size[0][2]:self.size[1][2]] = (np.transpose(sig3).dot(self.a[1]) + temp).reshape((self.size[1][0] * self.size[1][1],))

            self.state = 0
            self.grad /= self.x.shape[0]
            return self.grad

        # self.grad[0:self.size[0][2]] = (np.transpose(sig2).dot(self.a[0])).reshape((self.size[0][2],))
        #
        # temp = np.zeros_like(self.theta(params, 1))
        # if l != 0:
        #     temp[:, 1:] = self.theta(params, 1)[:, 1:] * l
        #
        # self.grad[self.size[0][2]:self.size[1][2]] = (np.transpose(sig3).dot(self.a[1])).reshape((self.size[1][0] * self.size[1][1],))

        # self.set_theta(self.grad, np.transpose(sig2).dot(self.a[0]), 0)
        # self.set_theta(self.grad, np.transpose(sig3).dot(self.a[1]), 1)

        # for i in range(1, end + 1):
        #     self.g[end - i] = self.g[end - i + 1].dot(self.theta(params, end - i + 1))  # (5000x10 10x26)
        #     self.g[end - i] = self.g[end - i][:, 1:] * sigmoid_grad(self.z[end - i])    # (5000x25 .* 5000x25)
        #
        # for i in range(0, end + 1):
        #     self.set_theta(self.grad, np.transpose(self.g[i]).dot(self.a[i]), i)

        # # regularization
        # if l != 0:
        #     for i in range(0, end):
        #         # only copy the first Columns
        #         # might be better than trying to 'delete' first column
        #         temp = np.zeros_like(self.params[i])
        #         temp[:, 0] = self.params[i][:, 0]
        #         self.g[i] += l * (self.params[i] - temp)


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

    def numerical_gradient(self, params, l=0, e=1e-4):

        grad = np.zeros_like(params)
        pertub = np.zeros_like(params)

        for i in range(0, len(params)):
            pertub[i] = e

            loss1 = self.cost_function(params - pertub, l)
            loss2 = self.cost_function(params + pertub, l)

            grad[i] = (loss2 - loss1) / (2.0 * e)
            pertub[i] = 0

        return grad

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
        if self.option['stochastic'] is True:
            return self.option['size']

        return self.x.shape[0]

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
        self.g = [np.zeros((1, 1))]

        for i in range(0, self.hidden_size() - 1):
            self.a.append(np.ones((self.train_size(), self.hidden(i) + 1)))    # Skipped
            self.z.append(np.ones((self.train_size(), self.hidden(i + 1))))
            self.g.append(np.zeros((1, 1)))

        self.g.append(np.zeros((1, 1)))
        self.a.append(np.ones((self.train_size(), self.last_hidden() + 1)))    # 5000x26
        self.z.append(np.ones((self.train_size(), self.output())))             # 5000x10

    def _generate_params(self):
        """allocate memory for params and initialize them with random numbers"""

        si = 0

        for i in range(0, len(self.size)):
            si += self.size[i][0] * self.size[i][1]
            self.size[i] += (si, )

        return np.random.uniform(0, 1, si)


if __name__ == "__main__":

    y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    y = format_labels(y)

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

    a = Perceptron(y, x, Options)

    print(a.size)
    print(a.params)
    # print(a.theta(0))
    print(a.theta(0).shape)
    # print(a.theta(1))
    print(a.theta(1).shape)


    # print(a.a[0].shape)
