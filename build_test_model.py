import numpy as np
from tensorflow.keras.layers import Dense, Dropout, GRU, SimpleRNN
from tensorflow.keras.models import Sequential
from utils.BASEDIR import BASEDIR

samples = 10000
# x_train = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(0, 20)] for _ in range(samples)])
# y_train = (((np.log2(((((x_train + 5.4) * 1.2905) - 1.94) ** 2)) - 2.5) ** .09) ** x_train + 25) * 1.25978

x_train = np.random.uniform(0, 10, (samples, 5, 1))
y_train = (x_train.take(axis=1, indices=4).take(axis=1, indices=0) * 2)
# y_train = (((x_train.take(axis=1, indices=0) + 2.9) / 2.4) + 1) ** 2



model = Sequential()
# model.add(GRU(256, input_shape=x_train.shape[1:]))
model.add(SimpleRNN(8, input_shape=x_train.shape[1:], return_sequences=True))
model.add(SimpleRNN(4))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=8, epochs=10, verbose=1, validation_split=0.2)
model.save('{}/examples/dense_model.h5'.format(BASEDIR))
print(model.predict([[[1], [1], [2], [3], [4]]]))
print('Saved!')
# exit()
