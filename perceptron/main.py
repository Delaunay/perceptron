# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import pandas as pd
import numpy as np

# from load_data import load_data
from cost_function import format_labels, Perceptron, add_bias
import cost_function as MPL

# structure of the NN
Options = {
    "stochastic": False,
    "size": 100,

    "structure": {
        "input": 3,
        "hidden": [5],
        "output": 3,
    }
}


print('=======================================================================')
print('                 Diagnostic                                            ')
print('=======================================================================')
print('* Load debugging data')

# load data
x = pd.read_csv('c:/class/test_data/xdebug.csv', header=None).values
y = pd.read_csv('c:/class/test_data/ydebug.csv', header=None).values
t1 = pd.read_csv('c:/class/test_data/t1debug.csv', header=None).values
t2 = pd.read_csv('c:/class/test_data/t2debug.csv', header=None).values

# super-hot thing
y = format_labels(np.transpose(y)[0])

# initialize the neural network with full data set
mpl = Perceptron(y, x, Options)

si = Options['structure']['input']
sh = Options['structure']['hidden'][0]
so = Options['structure']['output']

# Check size
print('* Params Size')
print(mpl.size)
print(t1.shape, t2.shape)
print('')

print('* Load debugging weights')
# set initial weight
mpl.params[0:(si + 1) * sh] = t1.reshape(((si + 1) * sh,))
mpl.params[(si + 1) * sh:(sh + 1) * so + (si + 1) * sh] = t2.reshape((so * (sh + 1),))

# compute cost
print('* Cost Function: ' + str(mpl.cost_function(mpl.params, 0)) + str(2.10095))
print('Diff: ' + str(mpl.cost_function(mpl.params, 0) - 2.10095))
print('')

# compute regularized cost
print('* Regularized Cost Function: ' + str(mpl.cost_function(mpl.params, 10)) + str(2.25231))
print('Diff: ' + str(mpl.cost_function(mpl.params, 10) - 2.25231))
print('')

t1g = pd.read_csv('c:/class/test_data/t1g.csv', header=None).values
t2g = pd.read_csv('c:/class/test_data/t2g.csv', header=None).values

print('* Gradient Checking')

g = mpl.gradient(mpl.params, -1)
num = mpl.numerical_gradient(mpl.params, 0)

gr = mpl.gradient(mpl.params, 10)
numr = mpl.numerical_gradient(mpl.params, 10)

grad = pd.DataFrame({'ag': g, 'anum': num, 'bgr': gr, 'bnumr': numr})
print(grad.head(10))

print('')
print('* Difference Should be Small:')
print(' Dif: ' + str(sum(grad['ag'] - grad['anum'])))
print('rDif: ' + str(sum(grad['bgr'] - grad['bnumr'])))

print('')
print('=======================================================================')
print('                 Real Data                                             ')
print('=======================================================================')
print('* Load data')

x = pd.read_csv('c:/class/test_data/x.csv', header=None).values
yl = pd.read_csv('c:/class/test_data/y.csv', header=None).values
t1 = pd.read_csv('c:/class/test_data/t1.csv', header=None).values
t2 = pd.read_csv('c:/class/test_data/t2.csv', header=None).values

# structure of the NN
Options = {
    "stochastic": False,
    "size": 100,

    "structure": {
        "input": 400,
        "hidden": [25],
        "output": 10,
    }
}

# super-hot thing
y = format_labels(np.transpose(yl)[0])

# initialize the neural network with full data set
mpl = Perceptron(y, x, Options)

si = Options['structure']['input']
sh = Options['structure']['hidden'][0]
so = Options['structure']['output']

print('Cost Function: ' + str(mpl.cost_function(mpl.params, 1)))
print('')

print('* Gradient Descent')
mpl.gradient_descent(0.5, 1, 5)

print('Accuracy : ' + str(mpl.accuracy(mpl.h, yl, 1)))

print('')
print('* LBFGS Optimization')
mpl.lbfgs()

print('Cost Function: ' + str(mpl.cost_function(mpl.params, 1)))
print('Accuracy : ' + str(mpl.accuracy(mpl.h, yl, 1)))

# print(yl.shape)
# print(hns(yl).shape)
# print('')
# yl = np.transpose(yl)[0]
# print(yl.shape)
# print(hns(yl).shape)
# print('')

# a = np.argmax(mpl.h, axis=1) + 1
#
# print(sum((MPL.hns(yl) == a).astype(float)))

# print(sum((MPL.predict_to_label(y) == a).astype(float)))


# p = np.argmax(mpl.h, axis=1) + 1
# yl = np.transpose(yl)[0]
# d = (p == yl).astype(int)
#
# print(d)
# print(mpl.accuracy())
# print(float(np.sum(d)) * 100.0/float(yl.shape[0]))


# p = np.argmax(mpl.h, axis=1) + 1
# print(p)

# d = (p == yl).astype(int)
#
# print(d)
# print('')
# print(mpl.accuracy())
# print(float(np.sum(d)) * 100.0/float(yl.shape[0]))

# print(mpl.h.shape)
# print(y.shape)
#

# def element_wise_equality(a, b):
#
#     c = np.zeros_like(a)
#
#     for i in range(0, len(a)):
#         for j in range(0, len(i)):
#
#             if a[i, j] == b[i, j]:
#                 c[i, j] = 1
#     return c


# def element_wise_equality(a, b):
#
#     c = np.zeros_like(a)
#
#     for i in range(0, len(a)):
#
#         if a[i] == b[i]:
#             c[i] = 1
#
#     return c
#
# print(element_wise_equality(mpl.h, y))



# t = np.equal(MPL.binarize(mpl.h), y)
# print(mpl.h)
# print(t)
# print(t.astype(int))



# print(pd.DataFrame(mpl.h))
# print(pd.DataFrame(mpl.y))
# print(pd.DataFrame(MPL.binarize(mpl.h)))

# # compute cost with reg
# print(str(mpl.cost_function(l=1))[0:8], 0.383770)
# print('Diff: ', mpl.cost_function(l=1.0) - 0.383770)
# print('')
#
# print(pd.DataFrame({
#     'g1': g[0:25*401],
#     'g1t': t1g.reshape((25*401,)),
#     'ng1': num[0:25*401]
# }).head(10))
#
# print('Grad2')
# print(pd.DataFrame({'g2': g[25 * 401:10 * 26 + 25 * 401], 'g2t': t2g.reshape((10 * 26,))}).head(10))
#

# print(mpl.gradient(mpl.params))


# print(mpl.accuracy((x, y)) * 100.0)
#
# mpl.stochastic_gradient((x, y), 0.00001, 100)
# # mpl.gradient_descent(1e-5, 1000)
# print(mpl.accuracy((x, y)) * 100.0)


# tx, ty, vx, vy, tex, tey = load_data("c:/class/")
# print()

# print()
# print(x)