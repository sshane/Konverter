import numpy as np
from tensorflow.keras.layers import Dense, Dropout, GRU, SimpleRNN, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from utils.BASEDIR import BASEDIR

samples = 5000
x_train = (np.random.rand(samples, 1) * 10)
# y_train = x_train.take(axis=1, indices=1) * 2
y_train = x_train * 2

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(amsgrad=True), loss='mse')
model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1, validation_split=0.2)

model.save('{}/examples/batch_norm.h5'.format(BASEDIR))
print(model.predict([[.5]]))
print('Saved!')
print(model.layers[0].get_weights()[0].shape)
print(model.layers[1].get_weights()[0].shape)
# exit()
