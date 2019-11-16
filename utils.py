
from typing import List, Iterable, Sized, Callable, Sequence

from scipy.optimize import fsolve
from math import isclose
import numpy as np


def modify_data(modifier, *data):
    for d in data:
        yield type(d)(modifier(x) for x in d)


def rolling_mean(data, N: int):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def mask_data(mask: Callable[[Iterable], List], keyarr: Iterable, *data: List[Iterable], modify_keyarr: bool = True,
              output_type_modifier: Callable = None):
    m: List = mask(keyarr)
    if modify_keyarr:
        result = (x for i, x in enumerate(keyarr) if m[i])
        if output_type_modifier is None:
            yield type(keyarr)(result)
        else:
            yield output_type_modifier(result)
    for arr in data:
        result = (x for i, x in enumerate(arr) if m[i])
        if output_type_modifier is None:
            yield type(arr)(result)
        else:
            yield output_type_modifier(result)


def index_mask_data(mask: Sequence, keyarr: Sequence, *data: List[Sequence], modify_keyarr: bool = True,
                    output_type_modifier: Callable = None):
    if modify_keyarr:
        result = (keyarr[i] for i in mask)
        if output_type_modifier is None:
            yield type(keyarr)(result)
        else:
            yield output_type_modifier(result)
        for arr in data:
            result = (arr[i] for i in mask)
            if output_type_modifier is None:
                yield type(arr)(result)
            else:
                yield output_type_modifier(result)


def closest_index(data: [Iterable, Sized], value):
    """ Find the index of the closest data point to value in the data set """
    data = sorted(data)
    return min(range(len(data)), key=lambda i: abs(data[i] - value))


def sort_together(keyarr: [Iterable, Sized], *arrs) -> List:
    """ sorts an arbitrary number of arrays the same way keyarr is sorted """
    idx = sorted(list(range(len(keyarr))), key=keyarr.__getitem__)
    yield list(map(keyarr.__getitem__, idx))
    for arr in arrs:
        yield list(map(arr.__getitem__, idx))


def dsolve(xdata, ydata, *args, **kwargs):
    """ Solve a given dataset for 0 by interleaving the data points with linear functions"""
    # need x to be sorted (monotonically increasing) for interleaving
    xdata, ydata = sort_together(xdata, ydata)

    # helper function that connects data points with a linear function
    def f(x):
        if isinstance(x, Iterable):
            return [f(y) for y in x]

        idx = closest_index(xdata, x)
        x1, y1 = xdata[idx], ydata[idx]

        if x >= x1 and idx != len(xdata) - 1 or idx == 0:
            # connect to the right
            idx += 1
            x2, y2 = xdata[idx], ydata[idx]
        else:
            # connect to the left
            idx -= 1
            x2, y2 = xdata[idx], ydata[idx]

            # swap the points so they are in order from left to right
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        while isclose(x1, x2):
            # x1 and x2 must not be the same since we are dividing by their difference later
            # so go to the right or left even more to find an appropriate data point
            if x >= x1:
                # connect further to the right
                if idx == len(xdata) - 1:
                    # no more data, assume constant
                    return y1
                idx += 1
                x2, y2 = xdata[idx], ydata[idx]
            else:
                # connect further to the left
                if idx == 0:
                    # no more data, assume constant
                    return y1
                idx -= 1
                x2, y2 = xdata[idx], ydata[idx]

        # f(x) = dfdx * x + f0
        dfdx = (y2 - y1) / (x2 - x1)
        f0 = y1 - dfdx * x1

        return dfdx * x + f0

    return fsolve(f, *args, **kwargs)


def average_close_points(xdata, ydata, delta):
    pass