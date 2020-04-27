"""
  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter
"""

import numpy as np

wb = np.load('repo_files/examples/gru_model_weights.npz', allow_pickle=True)
w, b = wb['wb']

def gru(x, idx):
  states = [np.zeros(w[idx][0].shape[1], dtype=np.float32)]
  for ts in range(units):
      x_ = np.split(np.matmul(sample[ts], input_kernel) + input_bias, 3, axis=-1)
      recurrent = np.split(np.matmul(states[-1], recurrent_kernel) + recurrent_bias, 3, axis=-1)
      z = sigmoid(x_[0] + recurrent[0])

      states.append(z * states[-1] + (1 - z) * tanh(x_[2] + sigmoid(x_[1] + recurrent[1]) * recurrent[2]))

def simplernn(x, idx):
  states = [np.zeros(w[idx][0].shape[1], dtype=np.float32)]
  for step in range(x.shape[0]):
    states.append(np.tanh(np.dot(x[step], w[idx][0]) + np.dot(states[-1], w[idx][1]) + b[idx]))
  return np.array(states[1:])

def predict(x):
  l0 = simplernn(x, 0)
  l1 = simplernn(l0, 1)[-1]
  l2 = np.dot(l1, w[2]) + b[2]
  l2 = np.maximum(0, l2)
  l3 = np.dot(l2, w[3]) + b[3]
  return l3
