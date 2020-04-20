import numpy as np
from tensorflow.keras.layers import Dense, Dropout, GRU, SimpleRNN
from tensorflow.keras.models import Sequential, load_model
from utils.BASEDIR import BASEDIR

samples = 10
x_train = np.random.rand(samples, 4, 2)

model = load_model('{}/examples/dense_model.h5'.format(BASEDIR))
print(model.predict([[x_train[0]]]))

# exit()
