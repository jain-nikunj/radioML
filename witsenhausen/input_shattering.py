import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math as m

PI = tf.constant(m.pi)

np.random.seed(2018)
tf.set_random_seed(2018)
NUM_BINS = 6
XMIN = -1
XMAX = 1
BIN_WIDTH = (XMAX - XMIN) / (NUM_BINS - 2)
NUM_EPOCHS = 10000
LR = 0.001
BATCHSIZE = 1000

def generate_samples(x):
  """
  Wrapper which iterates through input
  """
  y = np.zeros((x.shape[0], NUM_BINS - 2))
  for i in range(x.shape[0]):
    y[i, :] = generate_sample(x[i, :])

  return y

def generate_sample(x):
  """
  Given an x, generates a corresponding output.
  """

  # Implement decision rule here

  output = np.zeros((1, NUM_BINS - 2))
  low = XMIN

  for i in range(NUM_BINS - 2):
    if low + i * BIN_WIDTH < x <= low + (i + 1) * BIN_WIDTH:
      output[0, i] = 1

  return output

def sinc(x):
  atzero = tf.divide(tf.sin(tf.multiply(PI, x)), 1)
  atother = tf.divide(tf.sin(tf.multiply(PI, x)), tf.multiply(PI, x))
  value = tf.where(tf.equal(x, 0), atzero, atother)
  return value

def create_network(num_bins):
  x = tf.placeholder(tf.float32, [None, 1])
  a = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32)
  b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

  y = tf.add(tf.matmul(x, a), b)

  input_mtx = tf.Variable(tf.ones((1, (num_bins - 2)), tf.float32))
  input_mtx = tf.constant(1, shape=[1, num_bins - 2], dtype=tf.float32)
  input_bias = tf.reshape(
    tf.constant(
      [-(i+1) + (num_bins + 1)/2 for i in range(num_bins - 2)],
      dtype=tf.float32), (1, -1))

  bins = tf.add(tf.matmul(y, input_mtx), input_bias)
  activations = (sinc(bins))

  true_y = tf.placeholder(tf.float32, [None, num_bins - 2])
  cost = tf.losses.softmax_cross_entropy(true_y, activations)

  train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)


  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)
    acts, bins = sess.run([activations, bins], feed_dict = {x: [[0.5]]})

    for epoch in range(NUM_EPOCHS):
      batch_x = np.random.uniform(low=XMIN, high=XMAX, size=(BATCHSIZE, 1))
      batch_y = generate_samples(batch_x)
      batchcost, _, batchactivations = sess.run(
        [cost, train_op, activations], feed_dict = {x: batch_x, true_y: batch_y})

      if epoch % 100 == 0:
        print("Cost: " + str(batchcost))

    for batchx in np.linspace(XMIN, XMAX, 100):
      batchx = np.matrix(batchx).reshape((1, 1))
      batchy = generate_samples((batchx))
      act = sess.run(
        [activations], feed_dict = {x:batchx, true_y:batchy})
      print("Actual: " + str(batchy) + " Predicted: " + str(act))

    (n_a, n_b) = sess.run([a, b], feed_dict = {x: np.zeros((1, 1)),
                                               true_y: np.zeros((1, NUM_BINS - 2))})

    print("Learnt A: " + str(n_a))
    print("Learnt B: " + str(n_b))


create_network(NUM_BINS)
