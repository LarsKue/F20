
from typing import List, Iterable, Sized

from scipy.optimize import fsolve
from math import isclose
import numpy as np


def modify_data(modifier, *data):
    for d in data:
        yield type(d)(modifier(x) for x in d)


def closest_index(data: [Iterable, Sized], value):
    """ Find the index of the closest data point to value in the data set """
    data = sorted(data)
    return min(range(len(data)), key=lambda i: abs(data[i] - value))


def closest_indices(data: [Iterable, Sized], *values):
    for value in values:
        yield closest_index(data, value)


def sort_together(keyarr: [Iterable, Sized], *arrs) -> List:
    """ sorts an arbitrary number of arrays the same way keyarr is sorted """
    idx = sorted(list(range(len(keyarr))), key=keyarr.__getitem__)
    yield list(map(keyarr.__getitem__, idx))
    for arr in arrs:
        yield list(map(arr.__getitem__, idx))


def dsolve(xdata, ydata, *args, **kwargs):
    """ Solve a given dataset for 0 by interleaving the data points with linear functions"""
    # need x to be sorted for interleaving
    xdata, ydata = sort_together(xdata, ydata)

    # helper function that connects data points with a linear function
    def f(x):
        if isinstance(x, Iterable):
            # FIXME: idx comes out empty or all zeroes
            print(*x)
            print(type(x))
            idx = type(x)(list(closest_indices(xdata, *x)))
            print(idx)
            print(type(idx))
        else:
            idx = closest_index(xdata, x)

        print(idx)
        print(type(idx))
        x1, y1 = xdata[idx], ydata[idx]

        if x >= x1 or idx == 0:
            # connect to the right
            x2, y2 = xdata[idx + 1], ydata[idx + 1]
        else:
            # connect to the left
            x2, y2 = xdata[idx - 1], ydata[idx - 1]

            # swap the points so they are in order from left to right
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        if isclose(x1, x2):
            # x1 and x2 must not be the same since we are dividing by their difference later
            raise RuntimeError("dsolve cannot interleave datasets with duplicate xdata points")

        # f(x) = dfdx * x + f0
        dfdx = (y2 - y1) / (x2 - x1)
        f0 = y1 - dfdx * x1

        return dfdx * x + f0

    return fsolve(f, *args, **kwargs)


def average_close_points(xdata, ydata, delta):
    pass