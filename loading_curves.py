import sys

import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2


data_folder = "data/loading_curves/"


def get_data(filename, skip_rows=18, separator=","):
    with open(filename, "r") as f:
        for linenum, line in enumerate(f):
            if linenum < skip_rows:
                continue
            yield tuple(float(x) for x in line.strip().split(separator) if x)


def get_all_data():
    i = 4
    while True:
        filename = data_folder + "F{:04d}CH1.csv".format(i)
        try:
            yield zip(*list(get_data(filename)))
        except FileNotFoundError:
            break
        i += 1


def main(argv: list) -> int:
    print(list(get_data(data_folder + "F0004CH1.CSV")))

    for xdata, ydata in get_all_data():
        plt.plot(xdata, ydata)
        plt.show()

        print(xdata)
        print(ydata)
        print()
    return 0


if __name__ == "__main__":
    main(sys.argv)
