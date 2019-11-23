
import sys
import spectroscopy
import loading_curves

from utils import *

import numpy as np


def main(argv: list) -> int:
    spectroscopy.main(argv)
    loading_curves.main(argv)
    return 0


if __name__ == "__main__":
    main(sys.argv)
