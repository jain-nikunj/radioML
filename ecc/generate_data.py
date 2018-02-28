# Written by Nikunj Jain
import numpy as np
import commpy.channelcoding.convcode as cc

def generate_random_binary_sequence(n, p=0.5):
  """
  Return n bits of a random binary sequence, where each bit picked
  is a one with probability p.
  """
  seq = np.zeros(n)
  for i in range(n):
    seq[i] = 1 if (np.random.random() < p) else 0

  return seq

def convolution_encode(message_bits, generator_matrix, memory):
  """
  Given a sequence of input bits and a particular generator_matrix, along with
  a memory specification, generates the corresponding Trellis and then the
  convolution code.

  Returns encoded bits
  """

  trellis = cc.Trellis(memory, generator_matrix)
  coded_bits = cc.conv_encode(message_bits, trellis)

  return coded_bits

def random_conv_encode(n, generator_matrix, memory, p=0.5):
  """
  Generates a random sequence of n-bits, each Bernoulli(p) and then encodes
  them convolutionally. Returns the original sequence, and the convolutionally
  encoded one, using the generator matrix and memory.
  """

  message_bits = generate_random_binary_sequence(n, p)
  coded_bits = convolutional_encode(message_bits, generator_matrix, memory)

  return message_bits, coded_bits

