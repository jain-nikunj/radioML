import numpy as np
import cPickle
import matplotlib.pyplot as plt

def open_ds(location):
    f = open(location)
    ds = cPickle.load(f)
    return ds

def save_ds(dataset, location):
    cPickle.dump( dataset, file(location, 'wb' ) )

def main():
    ds = open_ds(location='datasets/radioml.dat')
    ds_trimmed = {}
    ds_trimmed[('BPSK', 18)] = ds[('BPSK', 18)]
    save_ds(dataset=ds_trimmed, location='datasets/bpsk.dat')




if __name__ == "__main__":
    main()