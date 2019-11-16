import sys

import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2

from copy import deepcopy

from utils import *

data_folder = "data/loading_curves/"
data_detuning = "data/detuning_coil_curves/"


def get_data(filename, skip_rows=18, separator=","):
    with open(filename, "r") as f:
        for linenum, line in enumerate(f):
            if linenum < skip_rows:
                continue
            yield tuple(float(x) for x in line.strip().split(separator) if x)


def get_all_data(directory_name, namerange, ignore=None):
    for i in namerange:
        if ignore and i in ignore:
            continue
        filename = directory_name + "F{:04d}CH1.csv".format(i)
        yield zip(*list(get_data(filename)))


def conv_volts_to_atomnumber(V_out, entry_in_detunings):
    powermeter_y_beam = ufloat(7.5, 0.5)  # mW
    powermeter_x_beam = ufloat(4.8, 0.4)
    powermeter_z_beam = ufloat(8.3, 0.5)

    p_powermeter = 2 * powermeter_y_beam + 2 * powermeter_x_beam + 2 * powermeter_z_beam

    def intens_0(x):  # We approximate the total power of all 6 laser beams to be:
        return (2 * x) / (np.pi * (0.2 ** 2))  # mW / cm^2

    # print("Intensity I0:", intens_0(p_powermeter))

    def detuning_calculator(x):
        return (2 * x) - 60 - (2 * 85)  # Mhz

    detunings = np.tile(
        abs(detuning_calculator(unp.uarray([109.75, 110.25, 110.75, 111.25, 111.75, 112.25], [0.03] * 6))), 2)

    # print(detunings)

    def gamma_sc(delta):
        gamma = 2 * np.pi * 6.07e6
        I_sat = 4.1  # mW / cm^2
        return (gamma / 2) * ((intens_0(p_powermeter) / I_sat) / (
                1 + (intens_0(p_powermeter) / I_sat) + 4 * ((delta ** 2) / gamma ** 2)))

    def wavelength_to_energy(l):
        return 6.62607015E-34 * 299792458 / l

    def conversion_to_atoms(V_out, entry_in_detunings):
        S = 1E6 / (1E6 + 50)
        T = 0.96
        G = ufloat(4.75E6, 4.75E6 * 0.05)  # V / A
        QE = ufloat(0.52, 0.015)  # A / W
        theta_omega = (np.pi * (25.4E-3 ** 2)) / (4 * np.pi * (150E-3 ** 2))

        return V_out / (QE * G * S * T * theta_omega * gamma_sc(detunings[entry_in_detunings]) * wavelength_to_energy(
            780E-9))

    return conversion_to_atoms(V_out, entry_in_detunings)
    # print(conversion_to_atoms(0.5, 0))

    # print("hier", detunings)
    #
    # return


def main(argv: list) -> int:
    # print(list(get_data(data_folder + "F0004CH1.CSV")))

    # for xdata, ydata in get_all_data(data_folder, range(4, 42 + 1), [18]):
    #     plt.plot(xdata, ydata)
    #     plt.show()
    #
    #     print(xdata)
    #     print(ydata)
    #     print()


    def loading_dgl(t, loading_rate, alpha, t0):
        return (loading_rate/alpha) * (1 - np.exp(-alpha*(t-t0)))

    def magnetic_field_gradient(i):
        return 1.1E-6 * (90 * i / (8.5 ** 2))  # i: current in Ampere, units: T/cm

    for xdata, ydata in get_all_data(data_detuning, range(4, 28 + 1)):
        x, y = zip(*list(get_data("data/detuning_coil_curves/F0028CH1.CSV")))
        y = np.array(y)
        y = np.where(y < 0.07993, 0.08505, y)

        # y = np.where(y > 0.0542, 0.05055, y)
        y = conv_volts_to_atomnumber(y, 0)
        # print([conv_volts_to_atomnumber(0.5,0),conv_volts_to_atomnumber(0.5,4)])
        # plt.plot(x, unp.nominal_values(y))
        xdata = np.array(xdata)

        ydata = np.array(ydata)
        ydata = conv_volts_to_atomnumber(ydata, 0)
        ydata = ydata - y
        # print(ydata)
        ydata = np.where(ydata < 0 , 0, ydata)



        mxdata, mydata = deepcopy(xdata), deepcopy(ydata)
        n = 50
        mydata = rolling_mean(mydata, n)

        @np.vectorize
        def mask(x):
            return 6.3 <= x <= 10

        mxdata, mydata = list(mxdata), list(mydata)

        mxdata, mydata = tuple(mask_data(mask, mask_rolling_mean(mxdata, n), mydata))
        popt, pcov = curve_fit(loading_dgl, mxdata, unp.nominal_values(mydata))
        plt.plot(mxdata, loading_dgl(mxdata, *popt))
        plt.plot(mxdata, unp.nominal_values(mydata), marker='.', linewidth=0)
        # plt.plot(xdata, unp.nominal_values(ydata), marker=".", linewidth=0)
        plt.xlabel("Time [s]")
        plt.ylabel("Number of Atoms")
        plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
