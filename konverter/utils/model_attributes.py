class Models:
  class _BaseModel:
    name = None
    alias = None

  class Sequential(_BaseModel):
    name = 'keras.Sequential'
    alias = 'sequential'

  class SequentialOld(_BaseModel):
    name = 'keras.models.Sequential'
    alias = 'sequential'

  class Unsupported(_BaseModel):
    pass


class Activations:
  """
    The class that contains the supported activations and any information we will need to generate models
      ex. activation in string format
    To add new activations, use the code_converter function and add them here!
  """

  class _BaseActivation:
    name = None
    alias = None
    string = None
    alpha = None
    needs_function = True

  class ReLU(_BaseActivation):
    name = 'keras.activations.relu'
    alias = 'relu'
    string = 'np.maximum(0, {})'
    needs_function = False

  class LeakyReLU(_BaseActivation):
    name = 'keras.layers.LeakyReLU'
    alias = 'LeakyReLU'
    string = 'np.where({0} > 0, {0}, {0} * {1})'
    alpha = 0.3  # default from tensorflow
    needs_function = False

  class Sigmoid(_BaseActivation):
    name = 'keras.activations.sigmoid'
    alias = 'sigmoid'
    string = 'def sigmoid(x):\n\treturn 1 / (1 + np.exp(-x))'

  class Softmax(_BaseActivation):
    name = 'keras.activations.softmax'
    alias = 'softmax'
    string = 'def softmax(x):\n\treturn np.exp(x) / np.sum(np.exp(x), axis=0)'

  class Tanh(_BaseActivation):
    name = 'keras.activations.tanh'
    alias = 'tanh'
    string = 'np.tanh({})'  # don't define a function if you don't want your string added to file as a function
    needs_function = False

  class Linear(_BaseActivation):
    name = 'keras.activations.linear'
    alias = 'linear'

  class Unsupported(_BaseActivation):  # propogated with act info and returned to Konverter if act is unsupported
    pass


class Layers:
  """
    The class that contains the supported layers and any information we will need to generate models
      ex. function in string format
    To add new layers, use the code_converter function and add them here!
  """

  class _BaseLayer:
    name = None
    alias = None
    string = None
    supported_activations = []
    needed_activations = []

  class Dense(_BaseLayer):
    name = 'keras.layers.Dense'
    alias = 'dense'
    supported_activations = [Activations.ReLU, Activations.Sigmoid, Activations.Softmax, Activations.Tanh, Activations.Linear, Activations.LeakyReLU]
    string = 'np.dot({}, w[{}]) + b[{}]'  # n0 is the previous layer, n1 is weight, n2 is bias

  class Dropout(_BaseLayer):
    name = 'keras.layers.Dropout'
    alias = 'dropout'

  class BatchNormalization(_BaseLayer):
    name = 'keras.layers.BatchNormalization'
    alias = 'batch_norm'
    string = 'def batch_norm(x, idx):\n' \
             '\tx = (x - mean[idx]) / np.sqrt(std[idx] + epsilon[idx])\n' \
             '\tx = gamma[idx] * x + beta[idx]\n\treturn x'

  class SimpleRNN(_BaseLayer):
    name = 'keras.layers.SimpleRNN'
    alias = 'SimpleRNN'
    supported_activations = [Activations.Tanh]
    string = 'def simplernn(x, idx):\n' \
             '\tstates = [np.zeros(w[idx][0].shape[1], dtype=np.float32)]\n' \
             '\tfor step in range({}):\n' \
             '\t\tstates.append(np.tanh(np.dot(x[step], w[idx][0]) + np.dot(states[-1], w[idx][1]) + b[idx]))\n' \
             '\treturn np.array(states[1:])'

  class GRU(_BaseLayer):
    name = 'keras.layers.GRU'
    alias = 'GRU'
    supported_activations = [Activations.Tanh, Activations.Sigmoid]
    needed_activations = [Activations.Tanh, Activations.Sigmoid]
    string = 'def gru(x, idx, units):\n' \
             '\tstates = [np.zeros(units, dtype=np.float32)]\n' \
             '\tfor step in range({}):\n' \
             '\t\tx_ = np.split(np.matmul(x[step], w[idx][0]) + b[idx][0], 3, axis=-1)\n' \
             '\t\trecurrent = np.split(np.matmul(states[-1], w[idx][1]) + b[idx][1], 3, axis=-1)\n' \
             '\t\tz = sigmoid(x_[0] + recurrent[0])\n' \
             '\t\tstates.append(z * states[-1] + (1 - z) * np.tanh(x_[2] + sigmoid(x_[1] + recurrent[1]) * recurrent[2]))\n' \
             '\treturn np.array(states[1:])'

  class Unsupported(_BaseLayer):  # propogated with layer info and returned to Konverter if layer is unsupported
    pass


class BaseModelInfo:
  supported = False
  input_shape = None  # this will need to be moved if we support functional models


class BaseLayerInfo:
  supported = False
  has_activation = False
  returns_sequences = False
  is_recurrent = False
  is_ignored = False

  activation = None
  weights = None
  biases = None

  gamma = None  # for BN
  beta = None
  mean = None
  std = None
  epsilon = None


def code_converter(indentation_spaces=2):
  """
  :param indentation_spaces: Enter the number of spaces for each 'indent' in your supplied code
  :return: A string representation of the input function
  """
  print('This converts code into a supported string format for Konverter.')
  print('Simply paste your function here, with no extra indentation. Each indent will be replaced with a \\t character.')
  code = input()
  return code.replace(' ' * indentation_spaces, '\t')


watermark = '"""\n\tGenerated using Konverter: https://github.com/ShaneSmiskol/Konverter\n"""\n\n'
