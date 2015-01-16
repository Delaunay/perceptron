# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import pandas as pd
import numpy as np

# from load_data import load_data
from cost_function import format_labels, Perceptron, add_bias

# load data
x = pd.read_csv('c:/class/test_data/x.csv', header=None).values
y = pd.read_csv('c:/class/test_data/y.csv', header=None).values
t1 = pd.read_csv('c:/class/test_data/t1.csv', header=None).values
t2 = pd.read_csv('c:/class/test_data/t2.csv', header=None).values

y = format_labels(np.transpose(y)[0])

# structure of the NN
structure = {
    "input": 400,
    "hidden": [25],
    "output": 10,
}

# initialize the neural network
mpl = Perceptron(y[0:100], x[0:100], structure)

# Check size
print(mpl.size)
print(t1.shape, t2.shape)
print('')

# set initial weight
mpl.params[0] = t1
mpl.params[1] = t2

# compute cost
print(str(mpl.cost_function())[0:8], 0.287629)
print('Diff: ', mpl.cost_function() - 0.287629)
print('')

# compute cost with reg
print(str(mpl.cost_function(l=1))[0:8], 0.383770)
print('Diff: ', mpl.cost_function(l=1.0) - 0.383770)
print('')

print(mpl.accuracy((x, y)) * 100.0)

mpl.stochastic_gradient((x, y), 0.00001, 100)
# mpl.gradient_descent(1e-5, 1000)
print(mpl.accuracy((x, y)) * 100.0)


# tx, ty, vx, vy, tex, tey = load_data("c:/class/")
# print()

# print()
# print(x)