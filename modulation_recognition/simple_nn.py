from helper_cyclo import *
import tensorflow as tf

np.random.seed(2018)

SNR_VALS = [18]
MOD_VALS = ['BPSK', 'QPSK']
n_train_factor = 0.80

# Set the following to true while debugging - throws away 90% of dataset
use_tenth=True

tf.logging.set_verbosity(tf.logging.INFO)

Xd = load_dataset("RML2016.10a_dict.dat")
snrs,mods = map(lambda j: sorted(list(
  set(map(lambda x: x[j], Xd.keys())))), [1,0])

mods_used = list(set(mods) & set(MOD_VALS))
snrs_used = list(set(snrs) & set(SNR_VALS))

X = []
lbl = []
for mod in mods_used:
    for snr in snrs_used:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

n_examples = X.shape[0]
n_train = int(n_examples * n_train_factor)

train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
Y_train = to_onehot(list(map(lambda x: mods_used.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods_used.index(lbl[x][0]), test_idx)))

classes = mods_used

if use_tenth:
  length = int(X_train.shape[0] * 0.1)
  X_train = X_train[np.arange(length)]
  Y_train = Y_train[np.arange(length)]

  length = int(X_test.shape[0] * 0.1)
  X_test = X_test[np.arange(length)]
  Y_test = Y_test[np.arange(length)]

signals_train = list(
  map(lambda data: data[0, :] + 1j*data[1, :], X_train)
) # Convert into a complex number from I and Q

signals_test = list(
  map(lambda data: data[0, :] + 1j*data[1, :], X_test)
) # Convert into a complex number from I and Q

signals_np = np.asarray(signals_train, np.complex64)
signals_np_test = np.asarray(signals_test, np.complex64)

# Create cyclostationary coefficients
cyclo_coeffs = np.asarray(wrapper_cyclo(signals_np), np.complex64)
cyclo_coeffs_test = np.asarray(wrapper_cyclo(signals_np_test), np.complex64)

# Stack up CS Coefficients with data signal
data_np_train = np.hstack((signals_np, cyclo_coeffs))
data_np_test = np.hstack((signals_np_test, cyclo_coeffs_test))


feature_columns = [
  tf.feature_column.numeric_column(key=str(i), dtype=tf.float64) for i in range(
    data_np_train.shape[1])
]

train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={str(i): data_np_train[:, i] for i in range(data_np_train.shape[1])},
  y=np.asarray(list(map(
    lambda one_hot_vector: np.argmax(one_hot_vector, axis=-1), Y_train))),
  shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={str(i): data_np_test[:, i] for i in range(data_np_test.shape[1])},
  y=np.asarray(list(map(
    lambda one_hot_vector: np.argmax(one_hot_vector, axis=-1), Y_test))),
  shuffle=False,
  num_epochs=1)

classifier = tf.estimator.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[5,5],
  n_classes=len(classes),
  optimizer=tf.train.ProximalAdagradOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.001
  )
)

classifier.train(
  input_fn=train_input_fn,
  steps=2000
)

classifier.evaluate(
  input_fn=test_input_fn,
  steps=2000
)

