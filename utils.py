
from typing import List, Iterable, Sized, Callable, Sequence, Union

from scipy.optimize import fsolve
from math import isclose
import numpy as np
from uncertainties import UFloat


def u_to_kg(m):
    return 1.66053904e-27 * m


def modify_data(modifier: Callable, *data):
    """ Apply a modifier function to every item in a dataset """
    for d in data:
        yield type(d)(modifier(x) for x in d)


def rolling_mean(data: Sequence, n: int):
    """ Computes the rolling mean of a data set over N data points """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def mask_rolling_mean(data: Sequence, n: int):
    """ Mask xdata for plotting with rolling mean ydata """
    return data[n // 2: -n // 2 + 1]


def mask_data(mask: Callable[[Iterable], Sequence], keyarr: Iterable, *data: List[Iterable], modify_keyarr: bool = True,
              output_type_modifier: Callable = None):
    """ Masks a dataset given a function that decides whether a value should be kept (based on that value)

    :param mask:    Function that takes an iterable data set and returns a sequence of the size of the data set,
                    with True or False values according to whether that data point should be kept

    :param keyarr:  The array which decides what data points should be kept

    :param data:    Other arrays that shall be masked in the same way

    :param modify_keyarr:   If False, the keyarray will be left as is and won't be yielded

    :param output_type_modifier:    For data set conversion between types, e.g. if you give this function a tuple but
                                    want a list, set output_type_modifier=list
                                    If set to None, data sets will be returned as their previous individual types

    :return:    The masked data sets
    """
    m: List = mask(keyarr)
    if modify_keyarr:
        result = list(x for i, x in enumerate(keyarr) if m[i])
        if output_type_modifier is None:
            yield type(keyarr)(result)
        else:
            yield output_type_modifier(result)
    for arr in data:
        result = list(x for i, x in enumerate(arr) if m[i])
        if output_type_modifier is None:
            yield type(arr)(result)
        else:
            yield output_type_modifier(result)


def index_mask_data(mask: Sequence, keyarr: Sequence, *data: List[Sequence], modify_keyarr: bool = True,
                    output_type_modifier: Callable = None):
    """ Masks a dataset given a sequence of indices, keeping only data points at those indices
        For additional info, see mask_data
    """
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


def interpolate_data(xdata: Sequence, ydata: Sequence) -> Callable:
    """ Returns a function that interpolates a dataset with linear functions """
    # need x to be sorted (monotonically increasing)
    xdata, ydata = sort_together(xdata, ydata)

    # the interpolating function which we will return
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

    return f


def dsolve(xdata: Sequence, ydata: Sequence, *args, **kwargs):
    """ Solve a given dataset for 0 by interpolating the data points with linear functions"""
    return fsolve(interpolate_data(xdata, ydata), *args, **kwargs)


def deviation(value: UFloat, lit_value: Union[float, UFloat]):
    if isinstance(lit_value, UFloat):
        sig = max(lit_value.std_dev, value.std_dev)
    else:
        sig = value.std_dev
    return abs(lit_value - value.nominal_value) / sig


def flatten(l: Iterable):
    """ Flattens any Iterable """
    for item in l:
        if isinstance(item, Iterable):
            yield from flatten(item)
            continue
        yield item
