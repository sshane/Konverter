import numpy as np
from tensorflow.keras.layers import Dense, Dropout, GRU, SimpleRNN
from tensorflow.keras.models import Sequential, load_model
from repo_utils.BASEDIR import BASEDIR

samples = 10
x_train = np.random.rand(samples, 4, 2)

model = load_model('{}/repo_files/examples/gru_model.h5'.format(BASEDIR))
print(model.predict([[[4, 4], [2.5, 1], [2, 2], [4, 4]]]))

# exit()
