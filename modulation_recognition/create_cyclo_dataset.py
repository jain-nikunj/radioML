from helper_cyclo import *
import pickle

datafile = "RML2016.10a_dict.dat"
outfile = "RML_Cyclocoeffs.dat"

ds = load_dataset(datafile)
cyclo_dataset = {}

for key, data in ds.items():
  complex_signal = data[:, 0, :] + 1j*data[:, 1, :]
  complex_coefficients = wrapper_cyclo(complex_signal)

  # Create tuple as (rows, 2, columns)
  new_shape = list(complex_coefficients.shape)
  new_shape = (new_shape[0], 2, new_shape[1])

  coeffs = np.zeros(new_shape)
  coeffs[:, 0, :] = complex_coefficients.real
  coeffs[:, 1, :] = complex_coefficients.imag

  cyclo_dataset[key] = coeffs

with open(outfile, 'wb') as f:
  pickle.dump(cyclo_dataset, f)



