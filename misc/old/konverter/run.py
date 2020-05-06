import os
from konverter import Konverter
from tensorflow.keras.models import load_model
from utils.BASEDIR import BASEDIR

os.chdir(BASEDIR)

model = load_model('examples/batch_norm.h5')
kon = Konverter()
kon.konvert(model, 'examples/batch_norm.py', 2, verbose=True)
