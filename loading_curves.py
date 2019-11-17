import sys

import math
import numpy as np
from math import erf
import scipy.special
from scipy.constants import Avogadro, k
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

from utils import *

data_folder = "data/loading_curves_reformatted/"
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

            omxdata, omydata = xdata, ydata

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

    for i in range(1, 4):
        plt.plot(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])), A[6*(i-1): (6*i)])
        plt.show()
    titles = np.array([r"$\alpha \ [\frac{1}{s}]$ vs. Detuning Frequency [MHz]",
                       r"Loss rate L $[\frac{1}{s}]$ vs. Detuning Frequency [MHz]"
                          , r"$N_{max} \ [-]$ vs. Detuning Frequency [MHz]"])
    ylabels = np.array([r"$\alpha \ [\frac{1}{s}]$", r"Loss rate L $[\frac{1}{s}]$", r"$N_{max} \ [-]$"])

    for z in range(1, 4):
        i = 1
        plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
                     all_fit_params[z - 1][6 * (i - 1): (6 * i)],
                     label="(9.0 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
        plt.xlabel("Detuning [MHz]")
        plt.ylabel(ylabels[z - 1])
        i = 2
        plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
                     all_fit_params[z - 1][6 * (i - 1): (6 * i)],
                     label="(9.5 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
        i = 3
        plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
                     all_fit_params[z - 1][6 * (i - 1): (6 * i)],
                     label="(10.0 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
        i = 4
        plt.errorbar(detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25])),
                     all_fit_params[z - 1][6 * (i - 1): (6 * i)],
                     label="(10.35 +/- 0.1)A", yerr=delta_all_fit_params[z - 1][6 * (i - 1): (6 * i)], fmt=".")
        plt.title(titles[z - 1])
        plt.legend()
        plt.show()
    def magnetic_field_gradient(current):
        return (1.1E-6 * (90 * current / (8.5 ** 2))) / (10 ** -6)  # i: current in Ampere, units: mikroT/cm

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
    # return 0
    titles_load_curve = np.array(
        ["10ms", "5ms", "3ms", "7ms", "9ms", "12ms", "11ms", "13ms", "14ms", "15ms", "16ms", "17ms", "18ms", "19ms",
         "20ms", "21ms", "22ms", "23ms",
         "24ms", "25ms", "26ms", "27ms", "28ms", "29ms", "30ms", "40ms", "50ms", "60ms", "70ms", "80ms", "90ms",
         "100ms"])

    mask_array_N0 = [(-1.14, -0.204), (-1.139, -0.0259), (-1.195, -0.259), (-1.193, -0.244), (-1.138, -0.094),
                     (-1.194, -0.204), (-1.194, -0.2046), (-1.63, -0.26), (-4.36, -0.63), (-4.47, -0.628),
                     (-4.259, -0.408), (-4.369, -0.738), (-4.47, -0.29), (-4.259, -0.628), (-4.369, -0.628),
                     (-4.479, -0.408), (-4.259, -0.408), (-4.479, -0.408), (-4.36, -0.628), (-4.36, -0.298),
                     (-4.479, -0.298), (-8.73, -1.47), (-8.95, -0.59), (-8.739, -0.59), (-8.739, -0.376),
                     (-8.739, -0.59), (-22.29, -1.39), (-22.84, -1.39), (-22.84, -3.04), (-23.0, -3.04),
                     (-22.84, -3.04), (-24.5, -1.39)]
    mask_array_Nmax = [(10.0295, 10.5798), (5.077, 5.796), (3.041, 4.08), (7.058, 7.77), (9.04, 9.86), (12.06, 12.83),
                       (11.075, 11.84), (12.99, 19.98), (14.22, 15.21), (15.108, 16.42), (16.2, 17.74), (17.0, 18.4),
                       (18.18, 19.5), (19.17, 20.61), (20.17, 22.04),
                       (21.16, 22.59), (22.04, 23.58), (22.15, 23.9),
                       (24.13, 25.67), (25.012, 26.99), (26.11, 28.31), (27.35, 29.55), (28.45, 31.09), (29.33, 31.75),
                       (30.43, 33.73), (40.56, 44.08), (50.88, 59.0), (60.23, 67.0), (70.68, 74.0), (81.69, 85.0),
                       (89.9, 94.0), (100.4, 105.0)]

    averages_N0 = []
    std_devs_N0 = []
    averages_Nmax = []
    std_devs_Nmax = []
    for i, (xdata, ydata) in enumerate(get_all_data(data_folder, range(4, 36))):
        a, b = mask_array_N0[i]
        c, d = mask_array_Nmax[i]
        x, y = zip(*list(get_data("data/loading_curves/F0011CH1.CSV")))
        y = np.array(y)
        y = np.mean(y[1000:1800])
        xdata = np.array(xdata)
        xdata = xdata / (10 ** -3)
        ydata = np.array(ydata) - y

        @np.vectorize
        def mask1(x):
            nonlocal a, b
            return a <= x <= b

        @np.vectorize
        def mask2(x):
            nonlocal c, d
            return c <= x <= d

        xdata, ydata = list(xdata), list(ydata)
        m1xdata, m1ydata = tuple(mask_data(mask1, xdata, ydata))
        m2xdata, m2ydata = tuple(mask_data(mask2, xdata, ydata))

        plt.plot(xdata, ydata , marker=".", linewidth=0)
        plt.title(titles_load_curve[i] + " Down time")
        plt.xlabel("time [ms]")
        plt.ylabel("Intensity [a.u.]")
        plt.show()
        averages_N0.append(np.mean(m1ydata))
        std_devs_N0.append(np.std(m1ydata))
        averages_Nmax.append(np.mean(m2ydata))
        std_devs_Nmax.append(np.std(m2ydata))
    # print(averages_N0)
    # print(std_devs_N0)
    uarray_Nmax = unp.uarray(averages_Nmax, std_devs_Nmax)
    uarray_N0 = unp.uarray(averages_N0, std_devs_N0)
    print(uarray_Nmax, uarray_N0)
    down_time = np.array(
        [10, 5, 3, 7, 9, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 50, 60, 70,
         80, 90, 100]) * 10 ** -3

    down_time, uarray_N0, uarray_Nmax = tuple(sort_together(down_time, uarray_N0, uarray_Nmax))
    uarray_N0 = np.array(uarray_N0)
    uarray_Nmax = np.array(uarray_Nmax)
    down_time = np.array(down_time)
    plt.errorbar(down_time, unp.nominal_values(uarray_Nmax / uarray_N0), marker=".",
                 yerr=unp.std_devs(uarray_Nmax / uarray_N0), label="data points")

    def fitfunc(x, temp):
        M = (85.4678 * (1.6605e-27))  # kg
        chi = np.sqrt(M / (k * temp)) * ((1.5e-3) / (x))

        return scipy.special.erf(chi) - ((2 / np.sqrt(np.pi)) * chi * np.exp(-(chi ** 2)))

    popt, pcov = curve_fit(fitfunc, down_time, unp.nominal_values(uarray_Nmax / uarray_N0),
                           sigma=unp.std_devs(uarray_Nmax / uarray_N0))

    plt.plot(down_time, fitfunc(down_time, *popt), label = r"Fit of $\frac{N(t)}{N_0} \ = \ erf(\chi) \ - \ \frac{2}{\sqrt{\pi}}\chi e^{-\chi^2}$")
    print("We get a MOT-Temperature of:", "(",popt[0] * 10 ** 6, "+-", np.sqrt(pcov[0][0]) * 10 ** 6,") micro Kelvin")
    plt.title("Fraction of Recaptured Atoms vs. Down Time")
    plt.xlabel("down time [s]")
    plt.ylabel(r"$\frac{N(t)}{N_0}$")
    plt.legend()
    plt.savefig("releaserecap.pdf", format="pdf")
    plt.show()
    return 0


if __name__ == "__main__":
    main(sys.argv)
