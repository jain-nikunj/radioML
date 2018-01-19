import numpy as np
from scipy import pi
import matplotlib.pyplot as plt
import pickle as cPickle
#Sine wave

N = 128

def get_sine_wave():
	x_sin = np.array([0.0 for i in range(N)])
	# print(x_sin)
	for i in range(N):
	  	# print("h")
	    x_sin[i] = np.sin(2.0*pi*i/16.0)

	plt.plot(x_sin)
	plt.title('Sine wave')
	plt.show()

	# y_sin = np.fft.fft(x_sin, N)
	# plt.plot(abs(y_sin))
	# plt.title('FFT sine wave')
	# plt.show()

	return x_sin

def get_bpsk_carrier():
	x = np.fromfile('gnuradio_dumps/bpsk_carrier', dtype = 'float32')
	x_bpsk_carrier = x[9000:9000+N]
	plt.plot(x_bpsk_carrier)
	plt.title('BPSK carrier')
	plt.show()

	# y_bpsk_carrier =  np.fft.fft(x_bpsk_carrier, N)
	# plt.plot(abs(y_bpsk_carrier))
	# plt.title('FFT BPSK carrier')
	# plt.show()

def get_qpsk_carrier():
	x = np.fromfile('gnuradio_dumps/qpsk_carrier', dtype = 'float32')
	x_qpsk_carrier = x[12000:12000+N]
	plt.plot(x_qpsk_carrier)
	plt.title('QPSK carrier')
	plt.show()


	# y_qpsk_carrier =  np.fft.fft(x_qpsk_carrier, N)
	# plt.plot(abs(y_qpsk_carrier))
	# plt.title('FFT QPSK carrier')
	# plt.show()

def get_bpsk():
	x = np.fromfile('gnuradio_dumps/bpsk', dtype = 'complex64')
	x_bpsk = x[9000:9000+N]
	plt.plot(x_bpsk.real)
	plt.plot(x_bpsk.imag)
	plt.title('BPSK')
	plt.show()



	# y_bpsk =  np.fft.fft(x_bpsk, N)
	# plt.plot(abs(y_bpsk))
	# plt.title('FFT BPSK')
	# plt.show()

def get_qpsk():
	x = np.fromfile('gnuradio_dumps/qpsk', dtype = 'complex64')
	x_qpsk = x[11000:11000+N]
	plt.plot(x_qpsk.real)
	plt.plot(x_qpsk.imag)
	plt.title('QPSK')
	plt.show()



	# y_qpsk =  np.fft.fft(x_bpsk, N)
	# plt.plot(abs(y_bqsk))
	# plt.title('FFT QPSK')
	# plt.show()

def load_dataset(location="../../datasets/radioml.dat"):
    f = open(location, "rb")
    ds = cPickle.load(f, encoding = 'latin-1')
    return ds


def get_from_dataset(dataset, key):
	"""Returns complex version of dataset[key][500]"""
	xr = dataset[key][500][0]
	xi = dataset[key][500][1]
	plt.plot(xr)
	plt.plot(xi)
	plt.title(key)
	plt.show()
	return xr

def main():
	
	
	x_sin = get_sine_wave()
	x_bpsk_carrier = get_bpsk_carrier()
	x_qpsk_carrier = get_qpsk_carrier()
	x_bpsk = get_bpsk()
	x_qpsk = get_qpsk()

	ds = load_dataset()
	x_amssb = get_from_dataset(dataset=ds, key=('AM-SSB', 18))
	x_amdsb = get_from_dataset(dataset=ds, key= ('AM-DSB', 18))
	x_gfsk = get_from_dataset(dataset=ds, key=('GFSK', 18))

if __name__ == "__main__":
    main()