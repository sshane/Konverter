import warnings
import os
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.layers import Dense, SimpleRNN, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session


def create_model(model_type):
  clear_session()
  samples = 2000
  epochs = 20
  if model_type == 'Dense':
    x_train = np.random.uniform(0, 10, (samples, 5))
    y_train = (np.mean(x_train, axis=1) ** 2) / 2  # half of squared mean of sample

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(1, activation='linear'))
  elif model_type == 'SimpleRNN':
    x_train = np.random.uniform(0, 10, (samples, 10, 4))
    y_train = (np.mean(x_train.take(axis=1, indices=8), axis=1) ** 2) / 2  # half of squared mean of sample's 8th index

    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape=x_train.shape[1:]))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
  elif model_type == 'GRU':
    x_train = np.random.uniform(0, 10, (samples, 10, 4))
    y_train = (np.mean(x_train.take(axis=1, indices=8), axis=1) ** 2) / 2  # half of squared mean of sample's 8th index

    model = Sequential()
    model.add(GRU(128, input_shape=x_train.shape[1:], return_sequences=True, implementation=2))
    model.add(GRU(64, return_sequences=True, implementation=2))
    model.add(GRU(32, implementation=2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
  else:
    raise Exception('Unknown model type: {}'.format(model_type))

  model.compile(optimizer='adam', loss='mse')
  model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)
  return model, x_train.shape
