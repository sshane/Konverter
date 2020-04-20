from utils.BASEDIR import BASEDIR
import numpy as np
import time
import os
from tensorflow import keras
from konverter import Konverter
from examples.dense_model import predict

os.chdir(BASEDIR)
model = keras.models.load_model('examples/dense_model.h5')
# Konverter(model, output_file='examples/dense_model', tab_spaces=2)  # creates the numpy model from the keras model

samples = np.random.uniform(0, 10, (1000, 5, 1)).astype('float32')
print('samples: {}\n'.format(len(samples)))

t = time.time()
model.predict(samples)
print('\nKeras model batch prediction time: {}s'.format(round(time.time() - t, 6)))

t = time.time()
predict(samples)
print('Konverted model batch prediction time: {}s'.format(round(time.time() - t, 6)))

print('-----')

t = time.time()
keras_preds = []
for i in samples:
  keras_preds.append(model.predict([[i]])[0][0])
print('Keras model single prediction time: {}s'.format(round(time.time() - t, 6)))

t = time.time()
konverter_preds = []
for i in samples:
  konverter_preds.append(predict(i)[0])
print('Konverted model single prediction time: {}s'.format(round(time.time() - t, 6)))


mae = np.mean(np.abs(np.array(keras_preds) - np.array(konverter_preds)))
mse = np.mean((np.array(keras_preds) - np.array(konverter_preds))**2)
print('\nkeras vs. konverted model (comparing models, lower is better):')
print('Mean absolute error for {} predictions: {}'.format(len(samples), mae))
print('Mean squared error for {} predictions: {}'.format(len(samples), mse))
