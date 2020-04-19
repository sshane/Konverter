import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from utils.BASEDIR import BASEDIR

x_train = np.array([[np.random.uniform(0, 20)] for _ in range(2000)])
y_train = ((np.log2(((((x_train + 5.4) * 1.2905) - 1.94) ** 2)) - 2.5) ** .09) ** x_train

model = Sequential()
model.add(Dense(128, input_shape=x_train.shape[1:], activation='relu'))  # needlessly large model for benchmark
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1)
model.save('{}/dense_model.h5'.format(BASEDIR))
