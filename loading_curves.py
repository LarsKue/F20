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
    for i in range(4, 42 + 1):
        if i == 19:
            continue
        filename = data_folder + "F{:04d}CH1.csv".format(i)
        yield zip(*list(get_data(filename)))


def conv_volts_to_atomnumber():

    def intens_0(p_powermeter):
        return (2 * p_powermeter) / (np.pi * ((2E-3) ** 2))

    def gamma_sc(delta):
        gamma = 2*np.pi*6.07E6
        I_sat = 4.1 #milivolts per squarecentimetre
        return (gamma/2) * ((intens_0(p_powermeter)/I_sat)/(1+(intens_0(p_powermeter)/I_sat) + 4 * ((delta ** 2)/gamma ** 2)))

    def wavelength_to_energy(lambda):
        return 6.62607015E-34 * 299792458 / lambda

    def conversion_to_atoms(V_out):
        S = 1E6 / (1E6 + 50)
        T = 0.96
        G = ufloat(4.75E6,4.75E6 * 0.05) # Volts/Ampere
        QE ufloat(0.52,0.015) # Ampere/Watt
        theta_omega = (np.pi * ((25.4E-3) ** 2))/(4*np.pi*((150E-3) ** 2))

        return V_out / (QE * G * S * T * theta_omega * gamma_sc(delta) * E_nu)

    return


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
