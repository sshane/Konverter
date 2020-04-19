import numpy as np
from utils.BASEDIR import BASEDIR
from tensorflow import keras
from utils.support_classes import SupportedAttrs, LayerInfo, ActivationStrings


class Konverter:
  def __init__(self, model, model_name, output_file, tab_spaces):
    self.supported = SupportedAttrs()
    self.activation_strings = ActivationStrings()

    self.model = model
    self.model_name = model_name
    self.output_file = output_file
    self.tab_spaces = tab_spaces

    self.layers = []
    self.watermark = '"""\n  Created using Konverter: https://github.com/ShaneSmiskol/Konverter\n"""\n\n'
    self.is_model()

    self.get_layers()
    self.build_konverted_model()

  def build_konverted_model(self):
    tab = ' ' * self.tab_spaces
    self.message('Now building pure Python + NumPy model...')

    imports = ['import numpy as np']
    activations = []
    for activation_string in set(filter(None, [getattr(self.activation_strings, layer.activation, None) for layer in self.layers])):
      if activation_string is not None:
        activations.append(activation_string.replace('\n', f'\n{tab}'))

    wb = []
    model = ['def predict(x):']
    for idx, layer in enumerate(self.layers):
      wb.append(f'w{idx} = np.array({layer.weights.tolist()})')
      wb.append(f'b{idx} = np.array({layer.biases.tolist()})')

      prev_layer = 'x' if idx == 0 else f'l{idx - 1}'
      model.append(f'l{idx} = np.dot({prev_layer}, w{idx}) + b{idx}')

      if layer.activation == 'relu':
        model.append(f'l{idx} = relu(l{idx})')
    model.append(f'return l{len(self.layers) - 1}')

    imports = '\n'.join(imports)
    activations = '\n'.join(activations)
    wb = '\n'.join(wb)
    model = f'\n{tab}'.join(model)

    output = self.watermark + '\n\n'.join([imports, wb, activations, model]) + '\n'
    with open(f'{BASEDIR}/{self.output_file}', 'w') as f:
      f.write(output)

    self.message('Saved konverted model to {}/{}'.format(BASEDIR, self.output_file))

  def print_model_architecture(self):
    self.message('Successfully got model architecture!\n')
    print('Layers:\n-----')
    to_print = []
    for layer in self.layers:
      to_print.append('  ' + '\n  '.join([f'name: {layer.name}', f'nodes: {layer.weights.shape[1]}', f'activation: {layer.activation}']))
    print('\n-----\n'.join(to_print))

  def get_layers(self):
    for layer in self.model.layers:
      layer_info = self.get_layer_info(layer)
      if layer_info.supported:
        self.layers.append(layer_info)
      else:
        raise Exception('Layer {} not currently supported (check type or activation)'.format(layer_info.name))
    self.print_model_architecture()

  def get_layer_info(self, layer):
    layer_info = LayerInfo()
    layer_name = getattr(layer, '_keras_api_names') + (layer.name,)
    name_exists = [idx for idx, name in enumerate(layer_name) if name in self.supported.layers]
    if len(name_exists) > 0:
      layer_info.name = self.supported.layers[layer_name[name_exists[0]]]

    layer_activation = getattr(layer.activation, '_keras_api_names')

    if len(layer_activation) == 1:
      if layer_activation[0] in self.supported.activations:
        layer_info.activation = self.supported.activations[layer_activation[0]]

    if None not in [layer_info.name, layer_info.activation]:
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
  model = keras.models.load_model('{}/dense_model.h5'.format(BASEDIR))
  konverter = Konverter(model, model_name='dense_model', output_file='generated.py', tab_spaces=2)
