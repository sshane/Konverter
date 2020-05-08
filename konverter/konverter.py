from konverter.utils.model_attributes import Activations, Layers, watermark
from konverter.utils.konverter_support import KonverterSupport
import numpy as np
import importlib
import os

support = KonverterSupport()


class Konverter:
  def __init__(self, tf_verbose=False):
    if not tf_verbose:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  def konvert(self, input_model, output_file, indent_spaces=2, verbose=True, use_watermark=True):
    """
    :param input_model: Either the the location of your tf.keras .h5 model, or a preloaded model
    :param output_file: The desired path and name of the output files, will be automatically formatted if .py is the suffix
    :param indent_spaces: The number of spaces to use for indentation
    :param verbose: To print status messages from Konverter
    :param use_watermark: To prepend a watermark comment to model wrapper
    """
    self.input_model = input_model
    self.output_file = output_file
    self.indent = ' ' * indent_spaces
    self.verbose = verbose
    self.use_watermark = use_watermark

    self.layers = []
    self.start()

  def start(self):
    self.load_model()
    self.get_layers()
    if self.verbose:
      self.print_model_architecture()
    self.remove_unused_layers()
    self.parse_output_file()
    self.build_konverted_model()

  def build_konverted_model(self):
    self.print('\nNow building pure Python + NumPy model...')

    model_builder = {'imports': ['import numpy as np'],
                     'functions': [],
                     'load_weights': [],
                     'model': ['def predict(x):']}

    # add section to load model weights and biases
    model_builder['load_weights'].append(f'wb = np.load(\'{self.output_file}_weights.npz\', allow_pickle=True)')
    model_builder['load_weights'].append('w, b = wb[\'wb\']')
    if Layers.BatchNormalization.name in support.layer_names(self.layers):
      model_builder['load_weights'].append('gamma, beta, mean, std, epsilon = wb[\'gbmse\']')

    # builds the model and adds needed activation functions
    for idx, layer in enumerate(self.layers):
      prev_output = 'x' if idx == 0 else f'l{idx - 1}'

      # work on predict function
      if layer.name == Layers.Dense.name:
        model_line = f'l{idx} = {layer.string.format(prev_output, idx, idx)}'
        model_builder['model'].append(model_line)
        if layer.info.has_activation:
          if layer.info.activation.needs_function:
            lyr_w_act = f'l{idx} = {layer.info.activation.alias.lower()}(l{idx})'
          else:  # eg. tanh or relu
            lyr_w_act = layer.info.activation.string.lower().format(f'l{idx}')
            lyr_w_act = f'l{idx} = {lyr_w_act}'
          model_builder['model'].append(lyr_w_act)

      elif layer.info.is_recurrent:
        if layer.name == Layers.SimpleRNN.name:
          rnn_function = f'l{idx} = {layer.alias.lower()}({prev_output}, {idx})'
        elif layer.name == Layers.GRU.name:
          units = layer.info.weights[0].shape[1] // 3
          rnn_function = f'l{idx} = {layer.alias.lower()}({prev_output}, {idx}, {units})'
        else:
          raise Exception('Unknown recurrent layer type: {}'.format(layer.name))
        if not layer.info.returns_sequences:
          rnn_function += '[-1]'
        model_builder['model'].append(rnn_function)

      elif layer.name == Layers.BatchNormalization.name:
        model_line = f'l{idx} = {layer.alias.lower()}({prev_output}, {idx})'
        model_builder['model'].append(model_line)

      # work on functions: activations/layers
      if layer.info.activation is not None:
        if layer.info.activation.string is not None:
          if layer.info.activation.needs_function:  # don't add tanh/relu as a function
            model_builder['functions'].append(layer.info.activation.string)

      if layer.info.is_recurrent:  # recurrent layers are specially handled here, need to improve this
        func = layer.string.format(self.model_info.info.input_shape[1])
        model_builder['functions'].append(func)
        model_builder['functions'] += [act.string for act in layer.needed_activations if act.needs_function]

      if layer.name == Layers.BatchNormalization.name:
        model_builder['functions'].append(layer.string)

    model_builder['functions'] = set(model_builder['functions'])  # remove duplicates
    model_builder['model'].append(f'return l{len(self.layers) - 1}')

    self.save_model(model_builder)
    self.output_file = self.output_file.replace('\\', '/')

    self.print('\nSaved Konverted model!')
    self.print(f'Model wrapper: {self.output_file}.py\nWeights and biases file: {self.output_file}_weights.npz')
    self.print('\nMake sure to change the path inside the wrapper file to your weights if you move the file elsewhere.')
    if Activations.Softmax.name in support.model_activations(self.layers):
      self.print('Important: Since you are using Softmax, make sure that predictions are working correctly!')

  def save_model(self, model_builder):
    wb = []
    gbmse = []  # gamma, beta, mean, std, epsilon for batch normalization
    for layer in self.layers:
      w = layer.info.weights
      b = layer.info.biases
      wb.append([np.array(w), np.array(b)])

      # TODO: right now, if layer is not batch norm, gamma, beta, etc. will be saved anyway with None values
      # TODO: if layer is batch norm, the weights and biases will be saved with None values
      # TODO: need to only save what is needed, and fix above indexes to increment only with their layer type (batch norm or not)
      gamma = layer.info.gamma
      beta = layer.info.beta
      mean = layer.info.mean
      std = layer.info.std
      epsilon = layer.info.epsilon
      gbmse.append([np.array(gamma), np.array(beta), np.array(mean), np.array(std), np.array(epsilon)])

    wb = list(zip(*wb))
    gbmse = list(zip(*gbmse))
    kwargs = {'wb': wb}
    if Layers.BatchNormalization.name in support.layer_names(self.layers):
      kwargs['gbmse'] = gbmse
    np.savez_compressed('{}_weights'.format(self.output_file), **kwargs)

    output = ['\n'.join(model_builder['imports']),  # eg. import numpy as np
              '\n'.join(model_builder['load_weights']),  # loads weights and biases for model
              '\n\t'.join(model_builder['model'])]  # builds the predict function
    if len(model_builder['functions']) > 0:
      output.insert(2, '\n\n'.join(model_builder['functions']))  # houses the model helper functions

    output = '\n\n'.join(output) + '\n'  # now combine all sections

    if self.use_watermark:
      output = watermark + output

    with open(f'{self.output_file}.py', 'w') as f:
      f.write(output.replace('\t', self.indent))

  def remove_unused_layers(self):
    self.layers = [layer for layer in self.layers if layer.name not in support.unused_layers]

  def parse_output_file(self):
    if self.output_file[-3:] == '.py':
      self.output_file = self.output_file[:-3]

  def print_model_architecture(self):
    print('\nSuccessfully got model architecture!\n')
    print('Layers:\n-----')
    to_print = [[f'name: {layer.alias}'] for layer in self.layers]
    for idx, layer in enumerate(self.layers):
      if not layer.info.is_ignored:
        if layer.info.has_activation:
          to_print[idx].append(f'activation: {layer.info.activation.alias}')
        if layer.info.is_recurrent:
          to_print[idx].append(f'shape: {layer.info.weights[0].shape}')
        elif layer.info.weights is not None:
          to_print[idx].append(f'shape: {layer.info.weights.shape}')

      to_print[idx] = '  ' + '\n  '.join(to_print[idx])
    print('\n-----\n'.join(to_print))

  def get_layers(self):
    for layer in self.model.layers:
      layer = support.get_layer_info(layer)
      if layer.info.supported:
        self.layers.append(layer)
      else:
        raise Exception('Layer `{}` with activation `{}` not currently supported (check type or activation)'.format(layer.name, layer.info.activation.name))

  def load_model(self):
    if isinstance(self.input_model, str):
      if os.path.exists(self.input_model):
        models = importlib.import_module('tensorflow.keras.models')
        self.model = models.load_model(self.input_model)
      else:
        raise Exception('The supplied model file path doesn\'t exist!')
    else:
      self.model = self.input_model
    self.model_info = support.get_model_info(self.model)
    if "tensorflow.python.keras.engine" not in str(type(self.model)):
      raise Exception('Input model must be a Sequential tf.keras model, not {}'.format(type(self.model)))
    elif not self.model_info.info.supported:
      raise Exception('Model is `{}`, must be in {}'.format(self.model.name, support.attr_map(support.models, 'name')))

  def print(self, msg):
    if self.verbose:
      print(msg)
