import numpy as np
from tensorflow.keras.layers import Dense, Dropout, GRU, SimpleRNN, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from utils.BASEDIR import BASEDIR


def one_hot(idx):
  x = [0 for _ in range(3)]
  x[idx] = 1
  return x


samples = 10000
x_train = (np.random.rand(samples, 3) * 10)
y_train = np.array([one_hot(np.argmax(sample)) for sample in x_train])

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(lr=0.003, amsgrad=True), loss='categorical_crossentropy')
model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.2)

model.save('{}/examples/batch_norm.h5'.format(BASEDIR))
print(model.predict([[4.5, 4.5, 9]]).tolist())
print('Saved!')
# exit()
