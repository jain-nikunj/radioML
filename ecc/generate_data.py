# Written by Nikunj Jain
import numpy as np
import commpy.channelcoding.convcode as cc
from itertools import permutations

def generate_random_binary_sequence(n, p=0.5):
  """
  Return n bits of a random binary sequence, where each bit picked
  is a one with probability p.
  """
  seq = np.zeros(n)
  for i in range(n):
    seq[i] = 1 if (np.random.random() < p) else 0

  return seq

def convolutional_encode(message_bits, generator_matrix, memory):
  """
  Given a sequence of input bits and a particular generator_matrix, along with
  a memory specification, generates the corresponding Trellis and then the
  convolution code.

  Returns encoded bits
  """

  trellis = cc.Trellis(memory, generator_matrix)
  coded_bits = cc.conv_encode(message_bits, trellis)

  return coded_bits

def random_conv_encode(k, generator_matrix, memory, p=0.5):
  """
  Generates a random sequence of k-bits, each Bernoulli(p) and then encodes
  them convolutionally. Returns the original sequence, and the convolutionally
  encoded one, using the generator matrix and memory.
  """

  message_bits = generate_random_binary_sequence(k, p)
  coded_bits = convolutional_encode(message_bits, generator_matrix, memory)

  return message_bits, coded_bits

def gen_gmatrix(L, rate=0.5):
  """
  For a rate of 1/2 or of 1/3, and a given constraint length L, generates the
  commonly used G matrix. The default memory assumed is L-1.
  This function has the functionality equivalent of a lookup table.
  [Credits to pg. 789-790, Proakis and Salehi 2nd edition]

  In such conv coding, the assumed k is 1, and the n is k * rate. Further, such
  sequences are worked on one symbol at a time.
  """
  if rate == 1/3:
    dict_l_generator = {
      3: [5, 7, 7],
      4: [13, 15, 17],
      5: [25, 33, 37],
      6: [47, 53, 75],
      7: [133, 145, 175],
      8: [255, 331, 367],
      9: [557, 663, 711]
      }

  elif rate == 1/2:
    dict_l_generator = {
      3: [5, 7],
      4: [15, 17],
      5: [23, 35],
      6: [53, 75],
      7: [133, 171],
      8: [247, 371],
      9: [561, 753]
    }

  else:
    assert False, "This rate is currently not supported, {}".format(str(rate))

  return np.array([dict_l_generator[L]])

def entropy(p):
  """
  Return the Bernoulli entropy associated with p.
  """
  return -(p * np.log2(p) + (1-p) * np.log2(1-p))

def generate_encode_typical_set(L, k, rate=1/2, p=0.81):
  """
  Generates and encodes the typical set of sequences defined on the space by
  the particular value of p. The size of the typical set is approximately
  2^{nH(p)} which is equivalent to all the sequences when p is 0.5, but is
  considerably smaller when p is more biased.

  Returns an array of these sequences, and the corresponding encoded versions.
  """
  ent = entropy(p)
  generator_matrix = gen_gmatrix(L, rate)
  memory = np.array([L-1])

  base_seq = [0 for i in range(int(k * (1-p)))] + [1 for i in range(int(k * p))]
  typ_set = list(permutations(base_seq))

  encoded_seqs = [None for _ in range(len(typ_set))]

  for i, message_seq in enumerate(typ_set):
    encoded_seq = convolutional_encode(message_seq, generator_matrix, memory)
    encoded_seqs[i] = encoded_seq

  return typ_set, encoded_seqs

def generate_encode_random_sequences(L, k, n, rate=1/2, p=0.5):
  """
  Generates n random binary sequences of k bits each. Convolutionally encodes
  these using the corresponding generator matrix for L and the rate, encoding
  it one bit at a time to achieve the rate. The binary sequences are sampled
  as k iid Bernoulli(p) samples.

  Returns the random sequences, and the corresponding encoded ones.
  """
  message_seqs = [None for _ in range(n)]
  encoded_seqs = [None for _ in range(n)]

  generator_matrix = gen_gmatrix(L, rate)
  memory = np.array([L-1])

  for i in range(n):
    message_seq, encoded_seq = random_conv_encode(
      k, generator_matrix, memory, p)

    message_seqs[i] = (message_seq)
    encoded_seqs[i] = (encoded_seq)

  return message_seqs, encoded_seqs

def viterbi_decode_sequences(encoded_seqs, L, rate=1/2):
  """
  Given a list of convolutionally encoded sequences, uses the Viterbi
  algorithm on each element to decode it using a hard-decision boundary.
  The Trellis is generated as per the specified L and k.

  Returns a list of decoded elements.
  """
  decoded_seqs = [None for _ in range(len(encoded_seqs))]
  generator_matrix = gen_gmatrix(L, rate)
  memory = np.array([L-1])
  trellis = cc.Trellis(memory, generator_matrix)

  for i, encoded_seq in enumerate(encoded_seqs):
    decoded_seq = cc.viterbi_decode(encoded_seq.astype(float), trellis)
    decoded_seqs[i] = decoded_seq

  return decoded_seqs



