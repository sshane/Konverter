"""
  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter
"""

import numpy as np

wb = np.load('repo_files/examples/gru_model_weights.npz', allow_pickle=True)
w, b = wb['wb']

def gru(x, idx, units):
  states = [np.zeros(units, dtype=np.float32)]
  for step in range(69):
    x_ = np.split(np.matmul(x[step], w[idx][0]) + w[idx][2], 3, axis=-1)
    recurrent = np.split(np.matmul(states[-1], w[idx][1]) + w[idx][2], 3, axis=-1)
    z = sigmoid(x_[0] + recurrent[0])
    states.append(z * states[-1] + (1 - z) * np.tanh(x_[2] + sigmoid(x_[1] + recurrent[1]) * recurrent[2]))
  return np.array(states[1:])

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def predict(x):
  l0 = gru(x, 0, 8)
  l1 = gru(l0, 1, 4)[-1]
  l2 = np.dot(l1, w[2]) + b[2]
  return l2
