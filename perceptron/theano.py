# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'


import theano
import theano.tensor as T
import numpy as np

# # x = T.vector('x')
# # y = T.vector('y')
# # z = x + y
# # f = theano.function(inputs=[z, y], outputs=z)
#
# X = T.matrix('X')
# Z = (X ** 2).sum()
# grad_x = T.grad(Z, X)
#
# f = theano.function([X], grad_x)
#
# print f(np.ones((2, 2)))


x = T.scalar('x')
t = T.scalar('t')

a = theano.shared(1.0, name='a')
b = theano.shared(0.0, name='b')

y = a * x + b

err = (y - t) ** 2

grad_a, grad_b = T.grad(err, [a, b])

f = theano.function(inputs=[x, t], outputs=err, updates={a:a - 0.01 * grad_a, b:b - 0.01 * grad_b})