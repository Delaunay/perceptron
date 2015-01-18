# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import pandas as pd
import numpy as np
import timeit as tm

# from load_data import load_data
from cost_function import format_labels, Perceptron, add_bias


# def shuffle(x, y):
#
#     bx = np.zeros_like(x)
#     by = np.zeros_like(y)
#     ck = {}
#     idx = 0
#
#     for i in range(len(x)):
#
#         while True:
#
#             idx = np.random.uniform(0, len(x))
#
#             if idx not in ck:
#                 ck[idx] = 1
#                 break
#
#         ck[idx] = 1
#
#         bx[idx, :] = x[idx, :]
#         by[idx, :] = y[idx, :]
#
#     return bx, by


file_option = {
    'run_diag': True,
    'run_real': True
}

# Cut the training set into multiple set
# see the effect of adding training sample on local minima distribution Accuracy on Tset et VSet
# see the effect of adding hidden layer    on local minima distribution Accuracy on Tset et VSet
# see the effect of regularization

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
    print(mpl.size)

    # set initial weight
    # mpl.params[0:(si + 1) * sh] = t1.reshape(((si + 1) * sh,))
    # mpl.params[(si + 1) * sh:(sh + 1) * so + (si + 1) * sh] = t2.reshape((so * (sh + 1),))

    mpl.l = 0
    # compute cost
    print('* Cost Function: ' + str(mpl.cost_function(mpl.params, 0)) + ' ' + str(2.10095))
    print('Diff: ' + str(mpl.cost_function(mpl.params, 0) - 2.10095))
    print('')

    mpl.l = 10
    # compute regularized cost
    print('* Regularized Cost Function: ' + str(mpl.cost_function(mpl.params, 10)) + ' ' + str(2.25231))
    print('Diff: ' + str(mpl.cost_function(mpl.params, 10) - 2.25231))
    print('')

    print('* Gradient Checking')

    # g = mpl.usual_grad(mpl.params)  # mpl.gradient(mpl.params, 0)
    mpl.l = 0
    g = mpl.gradient(mpl.params, 0)
    num = mpl.numerical_gradient(mpl.params, 0)

    mpl.l = 10
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

    # super-hot thing
    y = format_labels(np.transpose(yl)[0])

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

    # initialize the neural network with full data set
    mpl = Perceptron(y, x, Options)

    print('Neurons: ' + str(len(mpl.params)))
    print(mpl.size)
    print('')

    print('Cost Function: ' + str(mpl.cost_function(mpl.params, 1)))
    print('Accuracy : ' + str(mpl.accuracy(mpl.h, yl, 1)))
    print('')

    print('* Gradient Descent')
    mpl.gradient_descent(0.5, 1, 5)

    print('Accuracy : ' + str(mpl.accuracy(mpl.h, yl, 1)))

    print('')
    print('* LBFGS Optimization')
    mpl.lbfgs(150)

    print('Cost Function: ' + str(mpl.cost_function(mpl.params, 1)))
    print('Accuracy : ' + str(mpl.accuracy(mpl.h, yl, 1)))
    print('')

    print('=======================================================================')
    print('                 Statistics                                            ')
    print('=======================================================================')

    # Change number of Iteration
    # Change number of Hidden Layer
    # Change number of Layer Size
    # Change regularization

    print('* Setting up environ')

    struct = [
        # [15],
        # [40],
        # [25],
        [15, 15],
        [25, 15],
        [40, 15]
    ]

    for i in struct:
        Options["structure"]["hidden"] = i
        mpl = Perceptron(y, x, Options)

        print(len(mpl.params))

    for i in struct:
        Options["structure"]["hidden"] = i

        ite = 200
        size = 1000
        reg = 1.0

        case_name = 'ite' + str(ite) + \
                    '.h' + str(len(Options['structure']['hidden'])) + \
                    '' + str(Options['structure']['hidden']) + '.s' + str(size) + \
                    '.l' + str(reg)

        folder_name = 'gen/'

        cost_hist = []
        params_hist = []

        csv_file = open(folder_name + 'data.' + case_name + '.csv', 'wb')

        print('* Starting Computing')
        print('')

        for i in range(0, size):
            mpl = Perceptron(y, x, Options)

            lb = lambda: mpl.lbfgs(ite, reg)
            time = tm.timeit(lb, number=1)

            cost_hist.append([mpl.cost_function(mpl.params, reg),
                              mpl.accuracy(mpl.h, yl, 1)])

            params_hist.append(mpl.params)

            csv_file.write(str(i) + ', ' +
                           str(cost_hist[i][0]) + ', ' +
                           str(cost_hist[i][1]) + ', ' +
                           str(np.linalg.norm(params_hist[i])) + ', ' +
                           str(time) + str('\n'))

            print(str(i) + '\t' + str(cost_hist[i]) + '\t' + str(time))

        csv_file.close()

        print('')
        print('* Statistics')
        cost_hist = np.array(cost_hist)
        print('Mean: ' + str(np.mean(cost_hist, axis=0)))
        print('Max : ' + str(np.max(cost_hist,  axis=0)))
        print('Min : ' + str(np.min(cost_hist,  axis=0)))
        print('Std : ' + str(np.std(cost_hist,  axis=0)))

    # pd.DataFrame(cost_hist).to_csv('data.csv')