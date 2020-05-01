"""
  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter
"""

import numpy as np

wb = np.load('examples/all_dense_weights.npz', allow_pickle=True)
w, b = wb['wb']



def predict(x):
  l0 = np.dot(x, w[0]) + b[0]
  l0 = np.maximum(0, l0)
  l1 = np.dot(l0, w[1]) + b[1]
  l1 = np.maximum(0, l1)
  l2 = np.dot(l1, w[2]) + b[2]
  l2 = np.maximum(0, l2)
  l3 = np.dot(l2, w[3]) + b[3]
  l3 = np.maximum(0, l3)
  l4 = np.dot(l3, w[4]) + b[4]
  l4 = np.maximum(0, l4)
  l5 = np.dot(l4, w[5]) + b[5]
  return l5