import numpy as np
from utils.BASEDIR import BASEDIR
from tensorflow import keras
from utils.support_classes import SupportedAttrs, LayerInfo, AttrStrings, Aliases
import os

supported = SupportedAttrs()
attr_strings = AttrStrings()
aliases = Aliases()


class Konverter:
  def __init__(self, model, output_file, tab_spaces):
    self.model = model
    self.output_file = os.path.join(BASEDIR, output_file)
    self.tab = ' ' * tab_spaces

    self.layers = []
    self.watermark = '"""\n  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter\n"""\n\n'
    self.is_model()

    self.get_layers()
    self.print_model_architecture()
    self.delete_unused_layers()
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

      if layer.has_activation:  # todo: check unneeded for now, since dropout is removed before this
        if layer.activation != aliases.activations.linear:
          activation = layer.activation.split('.')[-1].lower()
          model_builder['model'].append(f'l{idx} = {activation}(l{idx})')

        if layer.activation in attr_strings.activations:
          act_str = attr_strings.activations[layer.activation]
          model_builder['activations'].append(act_str.replace('\n', f'\n{self.tab}'))

    model_builder['activations'] = set(model_builder['activations'])  # remove duplicates
    model_builder['model'].append(f'return l{len(self.layers) - 1}')
    self.save_model(model_builder)
    self.message('Saved konverted model to {}.py and {}_weights.npz'.format(self.output_file, self.output_file))

  def save_model(self, model_builder):
    # save weights
    wb = list(zip(*[[np.array(layer.weights), np.array(layer.biases)] for layer in self.layers]))
    np.savez_compressed('{}_weights'.format(self.output_file), wb=wb)
    # save model loader/predictor
    output = [model_builder['imports'], model_builder['load_weights'], model_builder['activations']]
    output = ['\n'.join(section) for section in output] + [f'\n{self.tab}'.join(model_builder['model'])]
    with open(f'{self.output_file}.py', 'w') as f:
      f.write(self.watermark + '\n\n'.join(output) + '\n')

  def delete_unused_layers(self):
    self.layers = [layer for layer in self.layers if layer.name not in supported.layers_without_activations]

  def print_model_architecture(self):
    self.message('Successfully got model architecture!\n')
    print('Layers:\n-----')
    to_print = [[] for _ in range(len(self.layers))]
    for idx, layer in enumerate(self.layers):
      to_print[idx].append(f'name: {layer.name}')
      if layer.has_activation:
        to_print[idx].append(f'shape: {layer.weights.shape}')
        to_print[idx].append(f'activation: {layer.activation}')
      to_print[idx] = '  ' + '\n  '.join(to_print[idx])
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
    layer_info.name = getattr(layer, '_keras_api_names_v1')[0]  # assume only 1 name

    if layer_info.name not in supported.layers_without_activations:
      layer_info.has_activation = True
      activation = getattr(layer.activation, '_keras_api_names')
      if len(activation) == 1:
        layer_info.activation = activation[0]
      else:
        raise Exception('None or multiple activations?')

    if layer_info.name in supported.layers:
      if layer_info.has_activation:
        if layer_info.activation in supported.activations:
          layer_info.supported = True
      else:  # skip activation check if layer has no activation (eg. dropout)
        layer_info.supported = True

    if not layer_info.supported or not layer_info.has_activation:
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
    elif self.model.name not in supported.models:
      raise Exception('Model is `{}`, must be in {}'.format(self.model.name, supported.models))


if __name__ == '__main__':
  model = keras.models.load_model('{}/examples/dense_model.h5'.format(BASEDIR))
  konverter = Konverter(model, output_file='examples/dense_model', tab_spaces=2)
