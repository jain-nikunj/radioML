import numpy as np
import pickle
from tqdm import tqdm
from cyclocoefficients import *

def load_dataset(location):
  with open(location, "rb") as f:
    ds = pickle.load(f, encoding='latin-1')

  return ds

def wrapper_cyclo(data):
  """Presumes data is n datapoints"""
  n, _ = data.shape
  specs = np.zeros((n, 128), dtype='complex')

  for i in tqdm(range(n)):
    a, _ = cyclo_stationary(data[i, :])
    specs[i, :] = convert_to_1d(a)

  return specs

def singular_normalize(data_vector):
  """Normalize a single vector to norm 1"""
  norm = np.linalg.norm(data_vector)

  return data_vector / norm

def normalize(data):
  """Normalize an array of data vectors"""
  data = data.copy()
  n, _ = data.shape

  for i in range(n):
    data[i, :] = singular_normalize(data[i, :])

  return data

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

