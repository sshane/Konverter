import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from utils.BASEDIR import BASEDIR

x_train = np.array([np.random.uniform(0, 5) for _ in range(2000)])
y_train = ((x_train + 5.4) / 2.5) ** 1.5

model = Sequential()
model.add(Dense(32, input_shape=(1,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=8, epochs=5, verbose=1)
model.save('{}/dense_model.h5'.format(BASEDIR))