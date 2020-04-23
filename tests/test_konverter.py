import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)

from repo_utils.BASEDIR import BASEDIR
import numpy as np
import time
from tensorflow import keras
from konverter import Konverter
from tests.build_test_models import create_rnn_model
import importlib

os.chdir(BASEDIR)


def run_tests():
  ker_rnn_model, shape = create_rnn_model()
  Konverter(ker_rnn_model, 'tests/rnn_model', 2, verbose=True)
  kon_rnn_model = importlib.import_module('tests.rnn_model')

  samples = np.random.uniform(0, 10, (100, *shape[1:])).astype('float32')
  konverter_preds = []
  keras_preds = []
  for sample in samples:
    konverter_preds.append(kon_rnn_model.predict(sample)[0])
    keras_preds.append(ker_rnn_model.predict([[sample]])[0][0])
  mae = np.mean(np.abs(np.array(keras_preds) - np.array(konverter_preds)))
  mse = np.mean((np.array(keras_preds) - np.array(konverter_preds)) ** 2)
  print(mae)
  print(mse)
  assert mae < 1e-6
  assert mse < 1e-12
  print('Keras vs. Konverted model outputs test: successful')


if __name__ == '__main__':
  run_tests()
