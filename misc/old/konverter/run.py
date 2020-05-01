import os
from konverter import Konverter
from tensorflow.keras.models import load_model
from utils.BASEDIR import BASEDIR

os.chdir(BASEDIR)

model = load_model('examples/testme.h5')
kon = Konverter(model, 'examples/testme.py', 2, verbose=True)
