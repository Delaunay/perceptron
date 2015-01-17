# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import pandas as pd
import numpy as np

# from load_data import load_data
from cost_function import format_labels, Perceptron, add_bias

file_option = {
    'run_diag': True,
    'run_real': True
}

if file_option['run_diag']:

    print('=======================================================================')
    print('                 Diagnostic                                            ')
    print('=======================================================================')

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
    # print('* Params Size')
    # print(mpl.size)
    # print(t1.shape, t2.shape)
    print('')

    print('* Load debugging weights')
    # set initial weight
    mpl.params[0:(si + 1) * sh] = t1.reshape(((si + 1) * sh,))
    mpl.params[(si + 1) * sh:(sh + 1) * so + (si + 1) * sh] = t2.reshape((so * (sh + 1),))

    # compute cost
    print('* Cost Function: ' + str(mpl.cost_function(mpl.params, 0)) + ' ' + str(2.10095))
    print('Diff: ' + str(mpl.cost_function(mpl.params, 0) - 2.10095))
    print('')

    # compute regularized cost
    print('* Regularized Cost Function: ' + str(mpl.cost_function(mpl.params, 10)) + ' ' + str(2.25231))
    print('Diff: ' + str(mpl.cost_function(mpl.params, 10) - 2.25231))
    print('')

    print('* Gradient Checking')

    # g = mpl.usual_grad(mpl.params)  # mpl.gradient(mpl.params, 0)
    g = mpl.gradient(mpl.params, 0)
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

if file_option['run_real']:
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