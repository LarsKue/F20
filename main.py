
import sys
import spectroscopy
import loading_curves

from utils import *

import numpy as np


def main(argv: list) -> int:

    # x = np.linspace(-5, 5, 1000)
    #
    # x0 = np.array([-2, -1, 0, 1, 2, 3])
    #
    # print(list(closest_indices(x, *x0)))

    spectroscopy.main(argv)
    # loading_curves.main(argv)
    return 0


if __name__ == "__main__":
    main(sys.argv)
