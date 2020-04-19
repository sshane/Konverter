import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from utils.BASEDIR import BASEDIR

x_train = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(0, 20)] for _ in range(10000)])

y_train = (((np.log2(((((x_train + 5.4) * 1.2905) - 1.94) ** 2)) - 2.5) ** .09) ** x_train + 25) * 1.25978

model = Sequential()
model.add(Dense(256, input_shape=x_train.shape[1:], activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='linear'))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1, validation_split=0.5)
model.save('{}/examples/dense_model.h5'.format(BASEDIR))
print('Saved!')
# exit()
