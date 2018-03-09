from helper_cyclo import *
# import tensorflow as tf

SNR_VALS = [18]

ds = load_dataset("RML2016.10a_dict.dat")

# Load all the data corresponding to the desired SNR
data = [item for item in ds.items() if item[0][1] in SNR_VALS and item[0][0] == 'BPSK']

labels, data_signal = zip(*data)
signals = list(
  map(lambda data: data[:, 0, :] + 1j*data[:, 1, :], data_signal)
) # Convert into a complex number from I and Q

signals_np = np.asarray(signals, np.complex64)

# Start labeling from 1 to k
indices = [i+1 for i, label in enumerate(labels)]

signals_np_stacked = signals_np.reshape((-1, signals_np.shape[2]))

# Create an array with each index repeated corresponding to datapoint
labels_vector = np.asarray(
  [[i] * signals_np.shape[1] for i in indices]).flatten()

# Create cyclostationary coefficients
cyclo_coeffs = np.asarray(wrapper_cyclo(signals_np_stacked))

# Stack up CS Coefficients with data signal
data_np = np.hstack((signals_np_stacked, cyclo_coeffs))

data_tf = tf.convert_to_tensor(data_np, np.complex64)
labels_tf = tf.convert_to_tensor(labels_vector, np.float32)
labels_one_hot = tf.one_hot(labels_tf, depth=max(labels_tf))


