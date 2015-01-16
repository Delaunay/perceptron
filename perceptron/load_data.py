# -*- coding: utf-8 -*-
__author__ = 'Pierre Delaunay'

import cPickle
import gzip


def load_data(name):

    with gzip.open(name, 'rb') as f:
        train, valid, test = cPickle.load(f)

        tx, ty = train
        vx, vy = valid
        tex, tey = test

        return tx, ty, vx, vy, tex, tey

