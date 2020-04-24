import os
import numpy as np
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from repo_utils.BASEDIR import BASEDIR
os.chdir(BASEDIR)


def create_rnn_model():
  samples = 200
  x_train = np.random.uniform(0, 10, (samples, 10, 2))
  y_train = (x_train.take(axis=1, indices=9).take(axis=1, indices=0) / 2) ** 2

  model = Sequential()
  model.add(SimpleRNN(64, input_shape=x_train.shape[1:], return_sequences=True))
  model.add(SimpleRNN(32))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1, activation='linear'))

  model.compile(optimizer='adam', loss='mse')
  model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1, validation_split=0.2)
  return model, x_train.shape

# print(model.predict_on_batch(np.random.uniform(0, 10, (1, 10, 2))))
