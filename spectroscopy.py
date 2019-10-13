import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2

from typing import Callable, List


# add a linear function f(x) = a * x + b to the gaussian as an offset
def gaussian(x, amp, mu, sigma, a, b):
    return a * x + b + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def voltage_to_freq(v):
    dVdf = ufloat(1.9366, 0.0008) * 1e-10
    return v / dVdf


def mask_data(mask: Callable[[List], List], keyarr: List, *data: List[List], modify_keyarr: bool = True):
    m: List = mask(keyarr)
    if modify_keyarr:
        yield type(keyarr)(x for i, x in enumerate(keyarr) if m[i])
    for arr in data:
        yield type(arr)(x for i, x in enumerate(arr) if m[i])


def get_data(filename: str):
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            yield tuple(float(x) for x in line.split("\t"))


def calibration(plot=True):
    """ Frequency Calibration """
    calib_lims = [(-0.8, -0.41), (-0.41, -0.11), (0.13, 0.49), (0.63, 0.95)]
    inner_lims = [(-0.57, -0.505), (-0.31, -0.265), (0.30, 0.33), (0.77, 0.83)]
    calib_starting_vals = [[-0.3, -0.5, 0.05, -0.15, 0.6], [-0.4, -0.28, 0.05, 0.1, 0.6], [-0.3, 0.31, 0.05, 0, 0.6],
                           [-0.1, 0.8, 0.05, 0, 0.6]]

    data_out, data_in, pdh = zip(*list(get_data("data/fullspectrum.txt")))

    if plot:
        plt.plot(data_out, data_in)

    voltages = []

    for i in range(4):
        @np.vectorize
        def mask(x):
            return calib_lims[i][0] <= x <= calib_lims[i][1]

        mdata_out, mdata_in, mpdh = tuple(mask_data(mask, data_out, data_in, pdh))

        @np.vectorize
        def mask(x):
            return not (inner_lims[i][0] <= x <= inner_lims[i][1])

        mdata_out, mdata_in, mpdh = tuple(mask_data(mask, mdata_out, mdata_in, mpdh))

        popt, pcov = curve_fit(gaussian, mdata_out, mdata_in, p0=calib_starting_vals[i])
        x = np.linspace(mdata_out[0], mdata_out[-1], 10000)

        if plot:
            plt.plot(x, gaussian(x, *popt), color="orange", label="Calibration Fit")

        P0 = [ufloat(popt[j], np.sqrt(pcov[j][j])) for j in range(len(popt))]
        voltages.append(P0[1])

        print("P0 {}: {}".format(i, P0))

    Rb87F1F2 = 6.834682610904290e9  # Hz
    # Rb85F2F3 = 3.0357324390e9  # Hz

    delta_V = abs(voltages[3] - voltages[0])

    dVdf1 = delta_V / Rb87F1F2

    # delta_V = abs(f[2] - f[1])
    #
    # dVdf2 = delta_V / Rb85F2F3

    print("dVdf =", dVdf1)
    # print("dVdf2 =", dVdf2)

    if plot:
        plt.xlabel("Aux Out [V]")
        plt.ylabel("Aux In [V]")
        plt.title("Full Spectrum")
        plt.show()


def get_lorentz_data(plot=True):
    filenames = ["85F2", "85F2fine", "85F3", "85F3fine", "87F1", "87F1fine", "87F2", "87F2fine"]

    xlims = [(0.25, 0.28), None, (-0.29, -0.245), None, (0.65, 0.71), None, (0.15, 0.22), None]
    starting_values = [[-0.3, 0.25, 0.1, 0, 0.6], None, [-0.5, -0.25, 0.1, 0, 0.6], None, [-0.1, 0.7, 0.1, 0, 0.6],
                       None,
                       [-0.2, 0.2, 0.1, -0.15, 0.6], None]

    for filename, xlim, p0 in zip(filenames, xlims, starting_values):
        data_out, data_in, pdh = zip(*list(get_data("data/" + filename + ".txt")))

        if plot:
            plt.plot(data_out, data_in, label="Data")

        if xlim:
            @np.vectorize
            def mask(x):
                return x <= xlim[0] or x >= xlim[1]

            mdata_out, mdata_in, mpdh = tuple(mask_data(mask, data_out, data_in, pdh))

            popt, pcov = curve_fit(gaussian, mdata_out, mdata_in, p0=p0)
            x = np.linspace(data_out[0], data_out[-1], 10000)
            if plot:
                plt.plot(x, gaussian(x, *popt), color="orange", label="Gaussian Fit")

            lorentzdata = np.array(list(data_in)) - gaussian(np.array(list(data_out)), *popt)
            yield (data_out, tuple(lorentzdata))

        if plot:
            plt.xlabel("Aux Out [V]")
            plt.ylabel("Aux In [V]")
            plt.title(filename)
            plt.legend()
            plt.show()

        # if not xlim:
        #     yield (data_out, data_in)


def plot_lorentz_data(lorentz_data):
    for data_out, data_in in lorentz_data:
        plt.plot(data_out, data_in)
        plt.show()


def lorentzian(x: np.ndarray, x_0, gamma):
    return ((1 / (np.pi * gamma)) * ((gamma ** 2) / ((x - x_0) ** 2 + (gamma ** 2))))


def lorentzfit(lorentz_data): #masterdata-Sortierung: 0=85RbF2, 1=85RbF3, 2=87RbF1, 3=87RbF2
    masterdata_out = []
    for data_out, data_in in lorentz_data:
        data_out = np.array(data_out)
        masterdata_out.append(data_out)
    masterdata_in = []
    for data_out, data_in in lorentz_data:
        data_in = np.array(data_in)
        masterdata_in.append(data_in)
    #print(masterdata_out[3][674:687])

    # popt0, pcov0 = curve_fit(lorentzian, masterdata_out[1][392:410], masterdata_in[1][392:410], p0=[-0.2835,0.0001])
    # plt.plot(masterdata_out[1], masterdata_in[1])
    # plt.plot(masterdata_out[1][392:410], lorentzian(masterdata_out[1][392:410], *popt0))
    # plt.show()
    #print(data_out[392:410])



    # x = list(range(10))
    # y = list(reversed(range(3, 23, 2)))
    #
    # @np.vectorize
    # def mask(x):
    #    return x > 5
    #
    # print(x, y)
    #
    # x, y = tuple(mask_data(mask, x, y))
    #
    # print(x, y)






def main(argv: list) -> int:
    # calibration()
    lorentz_data = list(get_lorentz_data(plot=False))
    # plot_lorentz_data(lorentz_data)
    lorentzfit(lorentz_data)
    return 0


if __name__ == "__main__":
    main(sys.argv)
