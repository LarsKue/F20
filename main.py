
import sys
import spectroscopy
import loading_curves


def main(argv: list) -> int:
    # spectroscopy.main(argv)
    loading_curves.main(argv)
    return 0


if __name__ == "__main__":
    main(sys.argv)
