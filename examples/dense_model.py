"""
  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter
"""

import numpy as np

wb = np.load('examples/dense_model_weights.npz', allow_pickle=True)
w, b = wb['wb']

def simplernn(x, idx):
  states = [np.zeros(w[idx][0].shape[1], dtype=np.float32)]
  for step in range(x.shape[0]):
    states.append(np.tanh(np.dot(x[step], w[idx][0]) + np.dot(states[-1], w[idx][1]) + b[idx]))
  return np.array(states[1:])

def predict(x):
  l0 = simplernn(x, 0)
  l1 = simplernn(l0, 1)
  l2 = simplernn(l1, 2)[-1]
  l3 = np.dot(l2, w[3]) + b[3]
  l3 = np.maximum(0, l3)
  l4 = np.dot(l3, w[4]) + b[4]
  l4 = np.maximum(0, l4)
  l5 = np.dot(l4, w[5]) + b[5]
  return l5
