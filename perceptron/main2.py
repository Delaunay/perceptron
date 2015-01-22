# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

from math import *
import pandas as pd
import timeit as tm

# from load_data import load_data
from cost_function import format_labels, Perceptron
from helpers import *

x = pd.read_csv('c:/class/test_data/x.csv', header=None).values
yl = pd.read_csv('c:/class/test_data/y.csv', header=None).values
y = format_labels(np.transpose(yl)[0])

x, y, yl = shuffle(x, y, yl)

total = x.shape[0]
tsize = int(total * 0.8)
vsize = total - tsize

# cutting the data 60 - 40
train_set = {
    'x': x[0:tsize, :],
    'y': y[0:tsize, :],
    'yl': yl[0:tsize, :]
}

valid_set = {
    'x': x[tsize:, :],
    'y': y[tsize:, :],
    'yl': yl[tsize:, :]
}

tsyl = pd.DataFrame({
    'yl': hns(train_set['yl'])
})

vsyl = pd.DataFrame({
    'yl': hns(valid_set['yl'])
})

ts_rep = {}
vs_rep = {}

print(str('') + '\t' + str('Train') + '\t' + str('Valid'))
print(str(tsize) + '\t' + str(vsize))

for i in range(1, 10):
    ts_rep[i] = ((tsyl['yl'] == i).sum() * 100/tsize)
    vs_rep[i] = ((vsyl['yl'] == i).sum() * 100/vsize)

    print(str(i) + '\t' + str(ts_rep[i]) + '\t' + str(vs_rep[i]))


test = 10

# reg factor
reg_table = [
    0,
    1,
    5,
    10,
]

# iteration
ite_table = [
    50,
    100,
    150,
    200
]

# Struct to test
struct = [
    # [15],
    # [40],
    # [25],
    # [15, 15],
    [25, 15],
    # [40, 15]
]

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

print('')

for s in struct:

    Options["structure"]["hidden"] = s
    reg = 1.0     # reg_table[1]
    info = str()

    for i in range(0, test):

        mpl = Perceptron(train_set['y'], train_set['x'], Options)
        mpl.l = reg
        lb = lambda: mpl.lbfgs(ite_table[3])
        time = tm.timeit(lb, number=1)

        h1 = mpl.predict(train_set['x'], mpl.params)
        h2 = mpl.predict(valid_set['x'], mpl.params)

        info = str(i) + ', ' + \
               str(mpl.cost_function(mpl.params)) + ', ' +\
               str(mpl.accuracy(h1, train_set['yl'], 1)) + ', ' +\
               str(mpl.accuracy(h2, valid_set['yl'], 1)) + ', ' +\
               str(np.linalg.norm(mpl.params)) + ', ' +\
               str(time) + str('\n')

        print(info)