import sys

import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
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
        return (loading_rate / alpha) * (1 - np.exp(-alpha * (t - t0)))

    mask_array = [(6.3, 10), (3.9, 9.9), (0.39, 9.8), (0.29, 10.8), (0.244, 10.67), (3.634, 13.5), (0.2705, 10.4),
                  (-0.195, 10.16), (-0.0872, 10.16), (0.187, 10.7), (0.176, 10.86), (0.2903, 10.48), (0.2384, 10.83),
                  (0.29, 10.0684), (0.304, 10.5282), (0.452, 10.59), (0.344, 10.5), (0.3077, 9.67), (0.332, 10.31),
                  (0.332, 10.5), (0.2856, 10.79), (0.265, 10.369), (0.344, 10.32), (0.288, 10.34)]

    def fitparameter_getter():
        L = []
        delta_L = []
        A = []
        delta_A = []
        N_max = []
        delta_N_max = []

        for i, (xdata, ydata) in enumerate(get_all_data(data_detuning, range(4, 28))):  # oder 28 + 1
            a, b = mask_array[i]
            x, y = zip(*list(get_data("data/detuning_coil_curves/F0028CH1.CSV")))
            y = np.array(y)

            # y = np.where(y < 0.07993, 0.08505, y)
            y = np.mean(y[320:750])  # nominal Background value in volts
            # print(y)
            # y = np.where(y > 0.0542, 0.05055, y)
            y = conv_volts_to_atomnumber(y, 0)
            # print([conv_volts_to_atomnumber(0.5,0),conv_volts_to_atomnumber(0.5,4)])
            # plt.plot(x[320:750], [y]*len(x[320:750]), marker = ".", linewidth=0)
            xdata = np.array(xdata)

            ydata = np.array(ydata)
            ydata = conv_volts_to_atomnumber(ydata, 0)
            ydata = ydata - y
            # print(ydata)
            # ydata = np.where(ydata < 0, 0, ydata)

            omxdata, omydata = deepcopy(xdata), deepcopy(ydata)

            # n = 5
            # omydata = rolling_mean(omydata, n)

            @np.vectorize
            def mask(x):
                nonlocal a, b
                return a <= x <= b

            # print(a, b)
            mxdata, mydata = list(omxdata), list(omydata)
            # print(len(mxdata), len(mydata))
            mxdata, mydata = tuple(mask_data(mask, mxdata, mydata))
            popt, pcov = curve_fit(loading_dgl, mxdata, unp.nominal_values(mydata), maxfev=5000)
            # plt.plot(mxdata, loading_dgl(mxdata, *popt))
            # plt.plot(mxdata, unp.nominal_values(mydata), marker='.', linewidth=0)
            # # plt.plot(xdata, unp.nominal_values(ydata), marker=".", linewidth=0)
            # plt.xlabel("Time [s]")
            # plt.ylabel("Number of Atoms")
            # print("L=", popt[0]," alpha=", popt[1], "N_max=", popt[0]/popt[1])
            # plt.show()
            # print(popt[2])
            L.append(popt[0])
            delta_L.append(np.sqrt(pcov[0][0]))
            A.append(popt[1])
            delta_A.append(np.sqrt(pcov[1][1]))
            N_max.append(popt[0] / popt[1])
            delta_N_max.append(
                unp.std_devs(ufloat(popt[0], np.sqrt(pcov[0][0])) / ufloat(popt[1], np.sqrt(pcov[1][1]))))

        return L, delta_L, A, delta_A, N_max, delta_N_max

    L, delta_L, A, delta_A, N_max, delta_N_max = fitparameter_getter()
    print(A)
    print(delta_A)
    print(L)
    print(delta_L)
    print(N_max)

    all_fit_params = []
    all_fit_params.append(A)
    all_fit_params.append(L)
    all_fit_params.append(N_max)
    all_fit_params = np.array(all_fit_params)

    delta_all_fit_params = []
    delta_all_fit_params.append(delta_A)
    delta_all_fit_params.append(delta_L)
    delta_all_fit_params.append(delta_N_max)
    delta_all_fit_params = np.array(delta_all_fit_params)

    def detuning_calculator(x):
        return (2 * x) - 60 - (2 * 85)  # Mhz

    # for i in range(1, 4):
    #     plt.plot(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])), A[6*(i-1): (6*i)])
    #     plt.show()
    titles = np.array([r"$\alpha \ [\frac{1}{s}]$ vs. Detuning Frequency [MHz]",
                       r"Loss rate L $[\frac{1}{s}]$ vs. Detuning Frequency [MHz]"
                          , r"$N_{max} \ [-]$ vs. Detuning Frequency [MHz]"])
    ylabels = np.array([r"$\alpha \ [\frac{1}{s}]$", r"Loss rate L $[\frac{1}{s}]$", r"$N_{max} \ [-]$"])

    # for z in range(1, 4):
    #     i = 1
    #     plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
    #                  all_fit_params[z - 1][6 * (i - 1): (6 * i)],
    #                  label="(9.0 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
    #     plt.xlabel("Detuning [MHz]")
    #     plt.ylabel(ylabels[z - 1])
    #     i = 2
    #     plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
    #                  all_fit_params[z - 1][6 * (i - 1): (6 * i)],
    #                  label="(9.5 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
    #     i = 3
    #     plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
    #                  all_fit_params[z - 1][6 * (i - 1): (6 * i)],
    #                  label="(10.0 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
    #     i = 4
    #     plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
    #                  all_fit_params[z - 1][6 * (i - 1): (6 * i)],
    #                  label="(10.35 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
    #     plt.title(titles[z - 1])
    #     plt.legend()
    #     plt.show()
    def magnetic_field_gradient(current):
        return (1.1E-6 * (90 * current / (8.5 ** 2))) /(10**-6)  # i: current in Ampere, units: mikroT/cm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(detuning_calculator(np.array(
        [109.75, 110.25, 110.75, 111.25, 111.75, 112.25, 109.75, 110.25, 110.75, 111.25, 111.75, 112.25, 109.75, 110.25,
         110.75, 111.25, 111.75, 112.25, 109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
                   magnetic_field_gradient(np.array(
                       [9, 9, 9, 9, 9, 9, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 10, 10, 10, 10, 10, 10, 10.35, 10.35, 10.35,
                        10.35, 10.35, 10.35])), N_max)
    ax.plot(detuning_calculator(np.array(
        [109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
        magnetic_field_gradient(np.array(
            [9, 9, 9, 9, 9, 9])), N_max[0:6], label= "9.0 +/- 0.1 A")
    ax.plot(detuning_calculator(np.array(
        [109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
        magnetic_field_gradient(np.array(
            [9.5, 9.5, 9.5, 9.5, 9.5, 9.5])), N_max[6:12], label= "9.5 +/- 0.1 A")
    ax.plot(detuning_calculator(np.array(
        [109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
        magnetic_field_gradient(np.array(
            [10, 10, 10, 10, 10, 10])), N_max[12:18], label= "10 +/- 0.1 A")
    ax.plot(detuning_calculator(np.array(
        [109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
        magnetic_field_gradient(np.array(
            [10.35, 10.35, 10.35, 10.35, 10.35, 10.35])), N_max[18:24], label= "10.35 +/- 0.1 A")
    ax.set_xlabel('Detuning Frequency [MHz]')
    ax.set_ylabel(r'Magnetic Field Gradient [$\frac{\mu T}{cm}$]')
    ax.set_zlabel(r'$N_{max}$')
    plt.legend()
    plt.savefig("3dplot.pdf", format="pdf")
    plt.show()
    return 0


if __name__ == "__main__":
    main(sys.argv)
