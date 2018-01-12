import numpy as np
import cPickle
import matplotlib.pyplot as plt

def open_ds(location):
    f = open(location)
    ds = cPickle.load(f)
    return ds


def main():
    ds = open_ds(location = 'datasets/bpsk.dat')
    plt.plot(ds[('BPSK', 18)][25][0][:])
    plt.plot(ds[('BPSK', 18)][25][1][:])
    plt.show()





if __name__ == "__main__":
    main()