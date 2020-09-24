from konverter.utils.model_attributes import Activations, Layers, watermark
from konverter.utils.konverter_support import KonverterSupport
from konverter.utils.general import success, error, info, warning, COLORS
import numpy as np
import importlib
import os

support = KonverterSupport()


class Konverter:
  def __init__(self, input_model=None, output_file=None, indent=2, silent=False, no_watermark=False, tf_verbose=False):
    if not tf_verbose:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if input_model is not None:
      self._konvert(input_model,
                    output_file=output_file,
                    indent=indent,
                    silent=silent,
                    no_watermark=no_watermark)
    else:  # support deprecated behavior
      warning('Warning: Creating instances of Konverter() is deprecated, please use konverter.konvert(...) instead.')

  def _konvert(self, input_model, output_file=None, indent=2, silent=False, no_watermark=False):  # needs to be the same as class inputs above
    """
    See __main__.py for full documentation
    Extra documentation:
      :param input_model: Can by a model file path as well as a Sequential object
      :param output_file: .py, .h5, .hdf5 extensions will be automatically removed.
        if not supplied then will be saved to same directory as input_model
    """
    self.input_model = input_model
    self.output_file = output_file
    if not isinstance(indent, int):
      indent = 2
    self.indent = ' ' * indent
    self.verbose = not silent
    self.use_watermark = not no_watermark

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
    self.print('\n🔨 Now building pure Python + NumPy model...', 'info')

    model_builder = {'imports': ['import numpy as np'],
                     'functions': [],
                     'load_weights': [],
                     'model': ['def predict(x):',
                               'x = np.array(x, dtype=np.float32)']}  # convert input to float32

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
            if layer.info.activation.alpha is None:
              lyr_w_act = layer.info.activation.string.lower().format(f'l{idx}')
            else:  # custom alpha for leakyrelu
              lyr_w_act = layer.info.activation.string.lower().format(f'l{idx}', layer.info.activation.alpha)

            lyr_w_act = f'l{idx} = {lyr_w_act}'
          model_builder['model'].append(lyr_w_act)

      elif layer.info.is_recurrent:
        if layer.name == Layers.SimpleRNN.name:
          rnn_function = f'l{idx} = {layer.alias.lower()}({prev_output}, {idx})'
        elif layer.name == Layers.GRU.name:
          units = layer.info.weights[0].shape[1] // 3
          rnn_function = f'l{idx} = {layer.alias.lower()}({prev_output}, {idx}, {units})'
        else:
          raise Exception(error('Unknown recurrent layer type: {}'.format(layer.name), ret=True))
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
    self.print('🙌 Saved Konverted model!', 'success')
    self.print(f'Output model file: {self.output_file}.py', 'success')
    self.print(f'Weights and biases file: {self.output_file}_weights.npz', 'success')
    self.print('\n❗ Make sure to change the path inside the wrapper file to your weights if you move the file elsewhere.', 'info')
    if Activations.Softmax.name in support.model_activations(self.layers):
      self.print('😬 Important: Since you are using Softmax, make sure that predictions are working correctly!', 'warning')

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
    kwargs = {'wb': np.array(wb)}

    if Layers.BatchNormalization.name in support.layer_names(self.layers):
      kwargs['gbmse'] = np.array(gbmse, dtype=np.object)
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
    if self.output_file is None:  # user hasn't supplied output file path, use input file name in same dir
      dirname = os.path.dirname(self.input_model)
      basename = os.path.basename(self.input_model).split('.')
      self.print('\nNo output file specified, using the input file\'s name and same directory!', 'info')
    else:
      self.output_file = self.output_file.replace('\\', '/')
      dirname = os.path.dirname(self.output_file)
      basename = os.path.basename(self.output_file).split('.')
    # handle user entering (multiple) weird file extensions, but don't remove non-problematic exts
    #  and self.output_file is not None:  # only remove exts when user didn't enter output file (todo: append this to next line for this functionality, probably won't be needed tho)
    while len(basename) > 1 and basename[-1].lower() in ['py', 'h5', 'hdf5']:
      basename = basename[:-1]
    self.output_file = '{}/{}'.format(dirname, '.'.join(basename))

  def print_model_architecture(self):
    success('\nSuccessfully got model architecture! 😄\n')
    info('Layers:')
    to_print = [[COLORS.BASE.format(74) + f'name: {layer.alias}' + COLORS.ENDC] for layer in self.layers]
    max_len = 0
    indentation = '  '
    for idx, layer in enumerate(self.layers):
      if not layer.info.is_ignored:
        if layer.info.has_activation:
          to_print[idx].append(COLORS.BASE.format(205) + f'activation: {layer.info.activation.alias}' + COLORS.ENDC)
        if layer.info.is_recurrent:
          to_print[idx].append(COLORS.BASE.format(135) + f'shape: {layer.info.weights[0].shape}' + COLORS.ENDC)
        elif layer.info.weights is not None:
          to_print[idx].append(COLORS.BASE.format(135) + f'shape: {layer.info.weights.shape}' + COLORS.ENDC)
      max_len = max(max_len, max([len(l) for l in to_print[idx]]))
      to_print[idx] = indentation + '\n  '.join(to_print[idx])
    seperator = ''.join(['=' for _ in range(max_len - 20 + len(indentation))])  # account for color formatting taking up 20 chars
    seperator = COLORS.BASE.format(119) + '\n {}\n'.format(seperator) + COLORS.ENDC
    print(seperator.join(to_print))
    print(COLORS.ENDC, end='')

  def get_layers(self):
    for layer in self.model.layers:
      layer = support.get_layer_info(layer)
      if layer.info.supported:
        self.layers.append(layer)
      else:
        raise Exception(error('Layer `{}` with activation `{}` not currently supported (check type or activation)'.format(layer.name, layer.info.activation.name), ret=True))

  def load_model(self):
    if isinstance(self.input_model, str):
      self.input_model = self.input_model.replace('\\', '/')
      if os.path.exists(self.input_model):
        load_model = importlib.import_module('tensorflow.keras.models').load_model  # only import when needed

        # FIXME: for some reason tf 2 can't load models with LeakyReLU without custom_objects
        custom_leakyrelu = importlib.import_module('tensorflow.keras.layers').LeakyReLU
        self.model = load_model(self.input_model, custom_objects={'LeakyReLU': custom_leakyrelu})
      else:
        raise Exception(error('The supplied model file path doesn\'t exist!', ret=True))
    else:
      self.model = self.input_model
    self.model_info = support.get_model_info(self.model)
    if "tensorflow.python.keras.engine" not in str(type(self.model)):
      raise Exception(error('Input model must be a Sequential tf.keras model, not {}'.format(type(self.model)), ret=True))
    elif not self.model_info.info.supported:
      raise Exception(error('Model is `{}`, must be in {}'.format(self.model.name, support.attr_map(support.models, 'name')), ret=True))

  def print(self, msg, typ=None):
    if self.verbose:
      if typ is not None:
        globals()[typ](msg)
        return
      print(msg)

konvert = Konverter  # to allow import konverter, konverter.konvert(...)
