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


def conv_volts_to_atomnumber():
    def intens_0(p_powermeter):
        return (2 * p_powermeter) / (np.pi * (2e-3 ** 2))

    def gamma_sc(delta, p_powermeter):
        gamma = 2 * np.pi * 6.07e6
        I_sat = 4.1  # mV / cm^2
        return (gamma / 2) * ((intens_0(p_powermeter) / I_sat) / (
                1 + (intens_0(p_powermeter) / I_sat) + 4 * ((delta ** 2) / gamma ** 2)))

    def wavelength_to_energy(l):
        return 6.62607015e-34 * 299792458 / l

    def conversion_to_atoms(V_out, delta, E_nu):
        S = 1E6 / (1E6 + 50)
        T = 0.96
        G = ufloat(4.75e6, 4.75e6 * 0.05)  # V / A
        QE = ufloat(0.52, 0.015)  # A / W
        theta_omega = (np.pi * (25.4e-3 ** 2)) / (4 * np.pi * (150e-3 ** 2))

        # FIXME: Missing Argument in gamma_sc
        return V_out / (QE * G * S * T * theta_omega * gamma_sc(delta) * E_nu)

    def detuning_calculator(x):
        return 2 * x - 60 - 2 * 85

    detunings = detuning_calculator(np.array([109.75, 110.25, 110.75, 111.25, 111.75, 112.25]))

    print("Die Werte sind nicht durch Punkte getrennt, das sind einfach floats und beim Datentyp np.float64 lässt "
          "numpy scheinbar die Kommas weg in favor von den Punkten halt")

    print("hier", detuning_calculator(detunings))

    return


def main(argv: list) -> int:
    conv_volts_to_atomnumber()

    print(list(get_data(data_folder + "F0004CH1.CSV")))

    # for xdata, ydata in get_all_data(data_folder, range(4, 42 + 1), [18]):
    #     plt.plot(xdata, ydata)
    #     plt.show()
    #
    #     print(xdata)
    #     print(ydata)
    #     print()

    x, y = zip(*list(get_data("data/detuning_coil_curves/F0028CH1.CSV")))
    for xdata, ydata in get_all_data(data_detuning, range(4, 28 + 1)):
        y = np.array(y)
        y = np.where(y < 0.07, 0.0849, y)

        xdata = np.array(xdata)
        ydata = np.array(ydata)

        plt.plot(xdata, ydata - y)
        plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
