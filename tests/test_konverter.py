import warnings
import os
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import importlib
import tensorflow as tf
from packaging import version
from konverter import Konverter
from utils.BASEDIR import BASEDIR
from tests.build_test_models import create_model

os.chdir(BASEDIR)
konverter = Konverter()


def test_models():
  tests = {'Dense': {'max_mae': 1e-5, 'max_mse': 1e-11},
           'SimpleRNN': {'max_mae': 1e-4, 'max_mse': 1e-9},  # RNN models have higher error/variance for some reason
           'GRU': {'max_mae': 1e-4, 'max_mse': 1e-9}}
  if version.parse(tf.__version__) < version.parse('2.0.0a0'):
    del tests['GRU']
  samples = 1000
  for test in tests:
    print(f'\nCreating trained {test} model', flush=True)
    ker_model, data_shape = create_model(test)
    konverter.konvert(ker_model, f'tests/{test.lower()}_model', 2, verbose=False)
    kon_model = importlib.import_module(f'tests.{test.lower()}_model')

    x_train = np.random.uniform(0, 10, (samples, *data_shape[1:])).astype('float32')
    konverter_preds = []
    keras_preds = []

    print('Comparing model outputs\n', flush=True)
    for sample in x_train:
      konverter_preds.append(kon_model.predict(sample)[0])
      keras_preds.append(ker_model.predict_on_batch([[sample]])[0][0])

    mae = np.abs(np.subtract(keras_preds, konverter_preds)).mean()
    mse = np.square(np.subtract(keras_preds, konverter_preds)).mean()
    assert mae < tests[test]['max_mae']
    assert mse < tests[test]['max_mse']
    print(f'Mean absolute error: {mae} < {tests[test]["max_mae"]}')
    print(f'Mean squared error: {mse} < {tests[test]["max_mse"]}')
    print(f'{test} model passed!')
