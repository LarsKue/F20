import sys


def get_data():
    i = 4
    while True:
        try:
            with open("DataReleaseRecap/F{:4d}CH1.csv".format(i), "r") as f:
                print(i)
        except FileNotFoundError:
            print("asdwaw")
            break
        i += 1


def main(argv: list) -> int:
    get_data()
    return 0


if __name__ == "__main__":
    main(sys.argv)
