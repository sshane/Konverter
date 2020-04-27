from konverter import Konverter
from tensorflow.keras.models import load_model

model = load_model('repo_files/examples/gru_model.h5')
kon = Konverter(model, 'repo_files/examples/gru_model.py', 2, verbose=True)
