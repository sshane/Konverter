import numpy as np
from utils.BASEDIR import BASEDIR
from tensorflow import keras
from utils.support import SupportedAttrs, LayerInfo, AttrStrings, Aliases, watermark
import os

supported = SupportedAttrs()
attr_strings = AttrStrings()
aliases = Aliases()


class Konverter:
  def __init__(self, model, output_file, indent_spaces, use_watermark=True):
    """
    :param model: A preloaded Sequential Keras model
    :param output_file: The desired path and name of the output model files
    :param indent_spaces: The number of spaces to use for indentation
    :param use_watermark: To prepend a watermark comment to model wrapper
    """
    self.model = model
    self.output_file = os.path.join(BASEDIR, output_file)
    self.indent = ' ' * indent_spaces
    self.use_watermark = use_watermark

    self.layers = []
    self.start()

  def start(self):
    self.is_model()
    self.parse_output_file()
    self.get_layers()
    self.print_model_architecture()
    self.delete_unused_layers()
    self.build_konverted_model()

  def build_konverted_model(self):
    print('\nNow building pure Python + NumPy model...')

    model_builder = {'imports': ['import numpy as np'],
                     'functions': [],
                     'load_weights': [],
                     'model': ['def predict(x):']}

    # add section to load model weights and biases
    model_builder['load_weights'].append(f'wb = np.load(\'{self.output_file}_weights.npz\', allow_pickle=True)')
    model_builder['load_weights'].append('w, b = wb[\'wb\']')

    # builds the model and adds needed activation functions
    for idx, layer in enumerate(self.layers):
      prev_layer = self.layers[idx - 1] if idx > 0 else LayerInfo()
      prev_output = 'x' if idx == 0 else f'l{idx - 1}'

      if layer.name == aliases.layers.dense:
        model_builder['model'].append(f'l{idx} = np.dot({prev_output}, w[{idx}]) + b[{idx}]')
        if layer.activation != aliases.activations.linear:
          activation = layer.activation.split('.')[-1].lower()
          model_builder['model'].append(f'l{idx} = {activation}(l{idx})')
      elif layer.is_recurrent:
        rnn_function = f'l{idx} = simplernn({prev_output}, {idx})'
        if not layer.returns_sequences:
          rnn_function += '[-1]'
        model_builder['model'].append(rnn_function)

      if layer.has_activation:  # todo: check unneeded for now, since dropout is removed before this
        if layer.activation in attr_strings.activations:
          act_str = attr_strings.activations[layer.activation]
          model_builder['functions'].append(act_str)

        if layer.is_recurrent:
          lyr_str = attr_strings.layers[layer.name]
          model_builder['functions'].append(lyr_str)

    model_builder['functions'] = set(model_builder['functions'])  # remove duplicates
    model_builder['model'].append(f'return l{len(self.layers) - 1}')
    self.save_model(model_builder)
    print('\nSaved Konverted model!')
    self.output_file = self.output_file.replace('\\', '/')
    print(f'Model wrapper: {self.output_file}.py\nWeights and biases file: {self.output_file}_weights.npz')
    print('\nMake sure to change the path inside the wrapper file to your weights if you move the file elsewhere.')

  def save_model(self, model_builder):
    wb = list(zip(*[[np.array(layer.weights), np.array(layer.biases)] for layer in self.layers]))
    np.savez_compressed('{}_weights'.format(self.output_file), wb=wb)

    output = ['\n'.join(model_builder['imports']),
              '\n'.join(model_builder['load_weights']),
              '\n\n'.join(model_builder['functions']),
              '\n\t'.join(model_builder['model'])]
    output = '\n\n'.join(output) + '\n'  # combine all sections

    if self.use_watermark:
      output = watermark + output

    output = output.replace('\t', self.indent)
    with open(f'{self.output_file}.py', 'w') as f:
      f.write(output)

  def delete_unused_layers(self):
    self.layers = [layer for layer in self.layers if layer.name not in supported.layers_without_activations]

  def parse_output_file(self):
    if self.output_file[-3:] == '.py':
      self.output_file = self.output_file[:-3]

  def print_model_architecture(self):
    print('\nSuccessfully got model architecture!\n')
    print('Layers:\n-----')
    to_print = [[] for _ in range(len(self.layers))]
    for idx, layer in enumerate(self.layers):
      to_print[idx].append(f'name: {layer.name}')
      if layer.has_activation and not layer.is_ignored:
        to_print[idx].append(f'activation: {layer.activation}')
      if layer.name not in supported.ignored_layers:
        if layer.is_recurrent:
          to_print[idx].append(f'shape: {layer.weights[0].shape}')
        else:
          to_print[idx].append(f'shape: {layer.weights.shape}')

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
    layer_info.is_ignored = layer_info.name in supported.ignored_layers

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
      elif layer_info.is_ignored:  # skip activation check if layer has no activation (eg. dropout)
        layer_info.supported = True

    if not layer_info.supported or not layer_info.has_activation:
      return layer_info

    wb = layer.get_weights()
    if len(wb) == 2:
      weights, biases = wb
    elif len(wb) == 3 and layer_info.name in supported.recurrent_layers:
      *weights, biases = layer.get_weights()
      layer_info.returns_sequences = layer.return_sequences
      layer_info.is_recurrent = True
    else:
      raise Exception('Layer `{}` had an unsupported number of weights: {}'.format(layer_info.name, len(wb)))

    layer_info.weights = np.array(weights)
    layer_info.biases = np.array(biases)
    return layer_info

  def is_model(self):
    if str(type(self.model)) != "<class 'tensorflow.python.keras.engine.sequential.Sequential'>":
      raise Exception('Input for `model` must be a Sequential tf.keras model, not {}'.format(type(self.model)))
    elif self.model.name not in supported.models:
      raise Exception('Model is `{}`, must be in {}'.format(self.model.name, supported.models))


if __name__ == '__main__':
  model = keras.models.load_model('{}/examples/dense_model.h5'.format(BASEDIR))
  konverter = Konverter(model, output_file='examples/dense_model.py', indent_spaces=2)
