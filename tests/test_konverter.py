import os
import numpy as np
import importlib
from konverter import Konverter
from repo_utils.BASEDIR import BASEDIR
from tests.build_test_models import create_rnn_model

os.chdir(BASEDIR)


def test_rnn():
  print('\nCreating RNN model...')
  ker_rnn_model, shape = create_rnn_model()
  Konverter(ker_rnn_model, 'tests/rnn_model', 2, verbose=False)
  kon_rnn_model = importlib.import_module('tests.rnn_model')

  samples = np.random.uniform(0, 10, (10000, *shape[1:])).astype('float32')
  konverter_preds = []
  keras_preds = []

  print('Comparing models...')
  for sample in samples:
    konverter_preds.append(kon_rnn_model.predict(sample)[0])
    keras_preds.append(ker_rnn_model.predict_on_batch([[sample]])[0][0])

  mae = np.mean(np.abs(np.array(keras_preds) - np.array(konverter_preds)))
  mse = np.mean((np.array(keras_preds) - np.array(konverter_preds)) ** 2)
  mae_max = 1e-5
  mse_max = 1e-10
  assert mae < mae_max
  assert mse < mse_max
  print(f'Mean absolute error: {mae} < {mae_max}')
  print(f'Mean squared error: {mse} < {mse_max}')
  print('RNN model test passed!')
