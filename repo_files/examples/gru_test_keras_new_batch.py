import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from repo_utils.BASEDIR import BASEDIR
import os
import time
from tensorflow.python.keras import backend as K

os.chdir(BASEDIR)
model = load_model('{}/repo_files/examples/gru_model.h5'.format(BASEDIR))


def tanh(x):
  return np.tanh(x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def hard_sigmoid(x):
  return np.maximum(0, np.minimum(1, x * 0.2 + 0.5))


dense_w = np.array(model.layers[1].get_weights()[0])
dense_b = np.array(model.layers[1].get_weights()[1])

input_kernel = np.array(model.layers[0].get_weights()[0])
recurrent_kernel = np.array(model.layers[0].get_weights()[1])
bias = np.array(model.layers[0].get_weights()[2])
input_bias, recurrent_bias = tf.unstack(tf.convert_to_tensor(bias))
input_bias = np.array(input_bias)
recurrent_bias = np.array(recurrent_bias)


samples = np.random.uniform(0, 5, (30000, 4, 2))
# sample = np.array([[[4, 4], [1, 1], [2, 2], [4, 4]], [[4, 4], [2.5, 1], [2, 2], [4, 4]]])
units = 4

for _ in range(3):
  _t = time.time()
  for sample in samples:
    states = [np.zeros(units, dtype=np.float32)]
    for ts in range(units):
      x_ = np.split(np.matmul(sample[ts], input_kernel) + input_bias, 3, axis=-1)
      recurrent = np.split(np.matmul(states[-1], recurrent_kernel) + recurrent_bias, 3, axis=-1)
      z = sigmoid(x_[0] + recurrent[0])

      states.append(z * states[-1] + (1 - z) * tanh(x_[2] + sigmoid(x_[1] + recurrent[1]) * recurrent[2]))

  _t = time.time() - _t
  print(f'konverter time: {_t}')


# l0 = np.dot(h, dense_w) + dense_b
# print(l0.tolist())


# l0 = np.dot(outputs, dense_w) + dense_b
# print(l0.tolist())



# for t in range(len(sample)):
#   print(t)
#   Rt = np.dot(sample[t], input_kernel)
#   Rt += np.dot(Hts[-1], recurrent_kernel[0])
#   Rt += bias[0]
#   print(Rt.shape)
#
#   Zt = np.dot(sample[t], input_kernel)
#   Zt += np.dot(Hts[-1], recurrent_kernel[1])
#   Zt += bias[1]
#
#   Ht = np.dot(sample[t], input_kernel)
#   Ht += np.dot(Ht * Rt, recurrent_kernel[t]) + bias[0]
#   Ht = tanh(Ht)
#   print(Ht.shape)
#   Ht = Zt * Ht + (1 - Zt) * Ht
#   Y = np.dot(Ht, recurrent_kernel[t]) + bias
#   Hts.append(Ht)
#
#   # add = mulw + mulu + bias
#   # z = np.dot(sample[0], input_kernel)  #  + np.dot(Uz, h) + bz)




# def forward_prop_step(x_t, s_t1_prev):
#   z_t1 = sigmoid(recurrent_kernel[0].dot(x_e) + input_kernel[0].dot(s_t1_prev) + bias[0])
#   r_t1 = sigmoid(recurrent_kernel[1].dot(x_e) + input_kernel[1].dot(s_t1_prev) + bias[1])
#   c_t1 = tanh(recurrent_kernel[2].dot(x_e) + input_kernel[2].dot(s_t1_prev * r_t1) + bias[2])
#   s_t1 = (np.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
#
#   # o_t = T.nnet.softmax(V.dot(s_t1) + c)[0]
#
# forward_prop_step(1, np.zeros(4))


# timesteps = sample.shape[0]
# prev_s = np.zeros(4)
# for step in range(timesteps):
#   mulu = np.dot(sample[step], kernel)
#   mulw = np.dot(prev_s, recurrent_kernel)
#   add = mulw + mulu + bias
#   s = np.tanh(add)
#   mulv = np.dot(recurrent_kernel, s)
#   prev_s = np.array(s)
#
# l0 = np.dot(s, dense_w) + dense_b
# print(l0.tolist())


#
# weights = np.transpose(np.concatenate([np.transpose(input_matrix), recurrent_matrix], 1))
#
# gate_inputs = np.concatenate([sample, np.zeros(16)], 1)
# gate_inputs = np.matmul(gate_inputs, weights)
#
#
# gate_inputs = np.bias_add(gate_inputs, bias)
#
# output = tanh(gate_inputs)
# print(output)