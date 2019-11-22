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

from utils import *

data_folder = "data/spectroscopy/"


# add a linear function f(x) = a * x + b to the gaussian as an offset
def gaussian(x, amp, mu, sigma, a, b):
    return a * x + b + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


# chad function from wikipedia
def lorentzian(x, x0, gamma, amp, y0, a):
    return a * x + y0 + amp * gamma ** 2 / (((x - x0) ** 2 + gamma ** 2) * np.pi * gamma)


# virgin function from script
# def lorentzian(x, x0, gamma, amp, y0):
#     return y0 + (amp / (1 + (4 * (x - x0) ** 2 / gamma ** 2)))


def voltage_to_freq(v):
    dVdf = ufloat(1.9366, 0.0008) * 1e-10  # Rb87
    # dVdf = ufloat(1.9504, 0.0008) * 1e-10  # Rb85
    return v / dVdf


def get_data(filename: str):
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            yield tuple(float(x) for x in line.split("\t"))


def calibration(plot=True, log=True):
    """ Frequency Calibration """
    calib_lims = [(-0.8, -0.41), (-0.41, -0.11), (0.13, 0.49), (0.63, 0.95)]
    inner_lims = [(-0.57, -0.505), (-0.31, -0.265), (0.30, 0.33), (0.77, 0.83)]
    calib_starting_vals = [[-0.3, -0.5, 0.05, -0.15, 0.6], [-0.4, -0.28, 0.05, 0.1, 0.6], [-0.3, 0.31, 0.05, 0, 0.6],
                           [-0.1, 0.8, 0.05, 0, 0.6]]

    data_out, data_in, pdh = zip(*list(get_data(data_folder + "fullspectrum.txt")))

    if plot:
        plt.figure(figsize=(8, 6))
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

        if log:
            print("P0 {}: {}".format(i, P0))

    Rb87F1F2 = 6.834682610904290e9  # Hz
    Rb85F2F3 = 3.0357324390e9  # Hz

    delta_V = abs(voltages[3] - voltages[0])

    dVdf1 = delta_V / Rb87F1F2

    delta_V = abs(voltages[2] - voltages[1])

    if log:
        print("Test with literature value for Lorenz:", voltage_to_freq(delta_V) * 1e-9, "GHz")
        print("Deviation:", 100 * (voltage_to_freq(delta_V) - Rb85F2F3) / Rb85F2F3, "%")

    dVdf2 = delta_V / Rb85F2F3

    if log:
        print("dVdf =", dVdf1)
        # checking result
        print("dVdf2 =", dVdf2)

    if plot:
        plt.xlabel("Aux Out [V]")
        plt.ylabel("Aux In [V]")
        plt.title("Full Spectrum")
        plt.show()


def get_lorentz_data(plot=True, return_temperatures=False, log=True):
    filenames = ["85F2", "85F3", "87F1", "87F2"]

    xlims = [(0.25, 0.28), (-0.29, -0.245), (0.65, 0.71), (0.15, 0.22)]
    starting_values = [[-0.3, 0.25, 0.1, 0, 0.6], [-0.5, -0.25, 0.1, 0, 0.6], [-0.1, 0.7, 0.1, 0, 0.6],
                       [-0.2, 0.2, 0.1, -0.15, 0.6]]

    temperatures = []
    m85 = u_to_kg(84.911789738)
    m87 = u_to_kg(86.909180527)
    l = 780e-9  # m
    nu_0 = consts.c / l  # Hz

    masses = [m85, m85, m87, m87]

    for filename, xlim, p0, m in zip(filenames, xlims, starting_values, masses):
        data_out, data_in, pdh = zip(*list(get_data(data_folder + filename + ".txt")))

        if plot:
            plt.plot(data_out, data_in, label="Data")

        @np.vectorize
        def mask(x):
            return x <= xlim[0] or x >= xlim[1]

        mdata_out, mdata_in, mpdh = tuple(mask_data(mask, data_out, data_in, pdh))

        popt, pcov = curve_fit(gaussian, mdata_out, mdata_in, p0=p0)

        sigma = voltage_to_freq(ufloat(popt[2], np.sqrt(pcov[2][2])))

        T = sigma ** 2 * m * consts.c ** 2 / (nu_0 ** 2 * consts.k)

        if log:
            print("T =", T, "K")
        temperatures.append(T)

        lorentzdata = np.array(list(data_in)) - gaussian(np.array(list(data_out)), *popt)
        yield (data_out, tuple(lorentzdata))

        if plot:
            x = np.linspace(data_out[0], data_out[-1], 10000)
            plt.plot(x, gaussian(x, *popt), color="orange", label="Gaussian Fit")
            plt.xlabel("Aux Out [V]")
            plt.ylabel("Aux In [V]")
            plt.title(filename)
            plt.legend()
            plt.show()

    if return_temperatures:
        return temperatures


def get_hyperfine_data(plot=True):
    filenames = ["85F2fine", "85F3fine", "87F1fine", "87F2fine"]

    for filename in filenames:
        data_out, data_in, pdh = zip(*list(get_data(data_folder + filename + ".txt")))

        pdh = list(*modify_data(lambda x: -x, pdh))

        yield (data_out, data_in, pdh)

        if plot:
            fig, ax1 = plt.subplots(figsize=(10, 8))
            color = "tab:blue"
            ax1.set_xlabel("Aux Out [V]")
            ax1.set_ylabel("Aux In [V]", color=color)
            ax1.plot(data_out, data_in, color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            ax2 = ax1.twinx()

            color = "tab:green"
            ax2.set_ylabel("PDH [a.u.]", color=color)
            ax2.plot(data_out, pdh, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            # fig.tight_layout()
            plt.title(filename)
            plt.show()


def lorentzfit(lorentz_data, plot=True, return_gammas=False):
    for i in range(len(lorentz_data)):
        lorentz_data[i] = tuple(np.array(x) for x in lorentz_data[i])

    # these values are from the pdf
    mask_ranges = [[(0.252, 0.2575), None, (0.2700, 0.2747)],
                   [(-0.2856, -0.28158), (-0.2785, -0.2745), (-0.2535, -0.2471)],
                   [(0.66, 0.675), (0.6772, 0.6863), (0.7000, 0.7142)],
                   [None, (0.1551, 0.1665), (0.2022, 0.2119)]]

    starting_values = [[[0.2555, 0.003, 0.03, 0.015, 0], None, [0.272, 0.003, 0.004, 0.021, 0]],
                       [[-0.283, 0.003, 0.009, 0.019, 0], [-0.276, 0.003, 0.005, 0.0275, 0],
                        [-0.250, 0.003, 0.008, 0.02, 0]],
                       [[0.666, 0.005, 0.0065, 0.0028, 1], None, [0.706, 0.003, 0.008, 0.008, 1]],
                       [None, [0.160, 0.003, 0.0065, 0.0035, -0.5], [0.206, 0.003, 0.02, 0.005, 0]]]

    gammas = []

    for (data_out, data_in), mask_list, startval_list in zip(lorentz_data, mask_ranges, starting_values):
        for mask_range, p0 in zip(mask_list, startval_list):
            if not mask_range:
                continue

            @np.vectorize
            def mask(x):
                return mask_range[0] <= x <= mask_range[1]

            mdata_out, mdata_in = mask_data(mask, data_out, data_in, output_type_modifier=list)

            popt, pcov = curve_fit(lorentzian, mdata_out, mdata_in, p0=p0, maxfev=100000)

            gamma = ufloat(popt[1], np.sqrt(pcov[1][1]))

            gammas.append(voltage_to_freq(gamma) * 1e-6)

            if plot:
                plt.plot(data_out, data_in, marker=".")
                plt.xlim(plt.axes().get_xlim())
                plt.ylim(plt.axes().get_ylim())
                x = np.linspace(data_out[0], data_out[-1], 10000)
                plt.plot(x, lorentzian(x, *popt))

                plt.show()

    if return_gammas:
        return gammas


def hyperfine(plot=True, log=True):
    # some estimates will be shifted right or left in order not to find the wrong peak
    x0_arr = [[0.209736, 0.2124, 0.216923, 0.219873, 0.226917],
              [0.319, 0.327099, 0.332584, 0.339475, 0.352864],
              [0.654384, 0.6627, 0.672215, 0.679843, 0.6976],
              [0.07777, 0.0952475, 0.107574, 0.125365, 0.154338]]

    for x0, (data_out, data_in, pdh) in zip(x0_arr, get_hyperfine_data(plot=False)):
        mean_N = 20
        smooth_pdh = np.array(rolling_mean(pdh, mean_N))

        # find solutions for phd
        peaks = dsolve(np.array(mask_rolling_mean(data_out, mean_N)), smooth_pdh, x0=x0)

        # the uncertainty in the peaks can be estimated by the mean difference in data_out (i.e. mean x-step) and mean_N
        delta_peaks = np.mean(np.diff(data_out)) * mean_N

        peaks = unp.uarray(peaks, delta_peaks)
        if log:
            # print("peaks:", peaks)
            print("Separation in MHz:")
            for x in voltage_to_freq(np.diff(peaks)) * 1e-6:
                print("{}".format(x))
            # print("{}".format(voltage_to_freq(np.diff(peaks)) * 1e-6))

        if plot:
            fig, ax1 = plt.subplots(figsize=(10, 8))
            color = "tab:blue"
            ax1.set_xlabel("Aux Out [V]")
            ax1.set_ylabel("Aux In [V]", color=color)
            ax1.plot(data_out, data_in, color=color)
            # ax1.plot(mask_rolling_mean(data_out, mean_N), rolling_mean(data_in, mean_N), color="orange")
            # ax1.plot(mask_rolling_mean(data_out, mean_N), smooth_data_in, color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            ax2 = ax1.twinx()
            ax2.axhline(color="black", linestyle="--", alpha=0.4)

            for x0 in peaks:
                plt.axvline(x=x0.nominal_value, color="red", alpha=0.6)
                plt.axvline(x=x0.nominal_value - x0.std_dev, color="orange", alpha=0.6)
                plt.axvline(x=x0.nominal_value + x0.std_dev, color="orange", alpha=0.6)

            color = "tab:green"
            ax2.set_ylabel("PDH [a.u.]", color=color)
            # ax2.plot(data_out, pdh, color="orange")  # original pdh
            ax2.plot(mask_rolling_mean(data_out, mean_N), smooth_pdh, color=color)

            ax2.tick_params(axis="y", labelcolor=color)

            plt.show()

    values = [74, 137, 92, 131, 223, 305]
    lvalues = [63.40161, 120.64068, 72.218040, 156.947070, 229.16511, 266.650090]
    deltas = [5, 7, 10, 10, 10, 13]

    for v, l, d in zip(values, lvalues, deltas):
        sig = abs(v - l) / d
        print(sig)


"""

Separations:

    Note: I declared the very left peak in 87F1 to be the F'=0 peak, though this is controversial,
    since the peak to the right of the F'=1 peak is not visible in the spectrum then, although it
    should be much bigger than the F'=0 peak.
    
    84F2 -> F':
    
    Isotope     F       F'1     F'2     experimental value (MHz)    literature value (MHz)      sigma
    85          2       2       3       74 +/- 5                    63.401(61)                  2.1
    85          3       3       4       137 +/- 7                   120.640(68)                 2.3
    87          1       0       1       92 +/- 10                   72.2180(40)                 2.0
    87          1       1       2       131 +/- 10                  156.9470(70)                2.6
    87          1       0       2       223 +/- 10                  229.165(11)                 0.62
    87          2       2       3       305 +/- 13                  266.6500(90)                2.9
    
"""


def main(argv: list) -> int:
    calibration(plot=True, log=True)
    lorentz_data = list(get_lorentz_data(plot=True, log=True))
    lorentzfit(lorentz_data, plot=True)
    hyperfine(plot=True, log=True)
    return 0


if __name__ == "__main__":
    main(sys.argv)
