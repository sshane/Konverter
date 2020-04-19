import numpy as np
from utils.BASEDIR import BASEDIR
from tensorflow import keras
from utils.support_classes import SupportedAttrs, LayerInfo, ActivationFunctions
import os


class Konverter:
  def __init__(self, model, output_file, tab_spaces):
    self.supported = SupportedAttrs()
    self.activation_functions = ActivationFunctions()

    self.model = model
    self.output_file = os.path.join(BASEDIR, output_file)
    self.tab = ' ' * tab_spaces

    self.layers = []
    self.watermark = '"""\n  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter\n"""\n\n'
    self.is_model()

    self.get_layers()
    self.print_model_architecture()
    self.build_konverted_model()

  def build_konverted_model(self):
    self.message('Now building pure Python + NumPy model...')

    model_builder = {'imports': ['import numpy as np'],
                     'activations': [],
                     'load_weights': [],
                     'model': ['def predict(x):']}

    # add section to load model weights and biases
    model_builder['load_weights'].append(f'wb = np.load(\'{self.output_file}_weights.npz\', allow_pickle=True)')
    model_builder['load_weights'].append('w, b = wb[\'wb\']')

    # builds the model and adds needed activation functions
    for idx, layer in enumerate(self.layers):
      prev_layer = 'x' if idx == 0 else f'l{idx - 1}'
      model_builder['model'].append(f'l{idx} = np.dot({prev_layer}, w[{idx}]) + b[{idx}]')

      activation = layer.activation
      if activation != 'linear':
        if activation == 'tanh':
          model_builder['model'].append(f'l{idx} = np.tanh(l{idx})')
        else:
          model_builder['model'].append(f'l{idx} = {activation}(l{idx})')

      if activation in self.activation_functions.activations:
        act_str = self.activation_functions.activations[activation]
        model_builder['activations'].append(act_str.replace('\n', f'\n{self.tab}'))

    model_builder['model'].append(f'return l{len(self.layers) - 1}')
    self.save_model(model_builder)
    self.message('Saved konverted model to {}.py and {}_weights.npz'.format(self.output_file, self.output_file))

  def save_model(self, model_builder):
    # save weights
    wb = list(zip(*[[np.array(layer.weights), np.array(layer.biases)] for layer in self.layers]))
    np.savez_compressed('{}_weights'.format(self.output_file), wb=wb)
    # save model loader/predictor
    output = [model_builder['imports'], model_builder['load_weights'], set(model_builder['activations'])]
    output = ['\n'.join(section) for section in output] + [f'\n{self.tab}'.join(model_builder['model'])]
    with open(f'{self.output_file}.py', 'w') as f:
      f.write(self.watermark + '\n\n'.join(output) + '\n')

  def print_model_architecture(self):
    self.message('Successfully got model architecture!\n')
    print('Layers:\n-----')
    to_print = []
    for layer in self.layers:
      to_print.append('  ' + '\n  '.join([f'name: {layer.name}', f'shape: {layer.weights.shape}', f'activation: {layer.activation}']))
    print('\n-----\n'.join(to_print))

  def get_layers(self):
    for layer in self.model.layers:
      layer = self.get_layer_info(layer)
      if layer.supported:
        self.layers.append(layer)
      else:
        raise Exception('Layer `{}` with activation `{}` not currently supported (check type or activation)'.format(layer.name, layer.activation))

  def get_layer_info(self, layer):
    layer_info = LayerInfo()

    name = getattr(layer, '_keras_api_names')[0]  # assume only 1 name
    if name in self.supported.layers:
      layer_info.name = self.supported.layers[name]
    else:
      layer_info.name = name[0]

    activation = getattr(layer.activation, '_keras_api_names')
    if len(activation) == 1:
      if activation[0] in self.supported.activations:
        layer_info.activation = self.supported.activations[activation[0]]
      else:
        layer_info.activation = activation[0]
    else:
      raise Exception('Multiple activations?')

    if layer_info.name in self.supported.layers.values() and layer_info.activation in self.supported.activations.values():
      layer_info.supported = True
    else:
      return layer_info

    weights, biases = layer.get_weights()
    layer_info.weights = np.array(weights)
    layer_info.biases = np.array(biases)
    return layer_info

  def message(self, msg):
    print(f'\n- {msg}')

  def is_model(self):
    if str(type(self.model)) != "<class 'tensorflow.python.keras.engine.sequential.Sequential'>":
      raise Exception('Input for `model` must be a tf.keras model, not {}'.format(type(self.model)))
    elif self.model.name not in self.supported.models:
      raise Exception('Model is `{}`, must be in {}'.format(self.model.name, self.supported.models))


if __name__ == '__main__':
  model = keras.models.load_model('{}/examples/dense_model.h5'.format(BASEDIR))
  konverter = Konverter(model, output_file='examples/dense_model', tab_spaces=2)
