class Models:
  class Sequential:
    name = 'sequential'


class Activations:
  """
    The class that contains the activations in string format to be used in the output model file.
    To add new activations, use the code_converter function and add them here!
  """
  class ReLU:
    name = 'keras.activations.relu'
    string = 'def relu(x):\n\treturn np.maximum(0, x)'

  class Sigmoid:
    name = 'keras.activations.sigmoid'
    string = 'def sigmoid(x):\n\treturn 1 / (1 + np.exp(-x))'

  class Softmax:
    name = 'keras.activations.softmax'
    string = 'def softmax(x):\n\treturn np.exp(x) / np.sum(np.exp(x), axis=0)'

  class Tanh:
    name = 'keras.activations.tanh'
    string = 'def tanh(x):\n\treturn np.tanh(x)'

  class Linear:  # No activation, but is technically not `None`
    name = 'keras.activations.linear'
    string = None


class Layers:
  """
    The class that contains the layers in string format to be used in the output model file.
    To add new layers, use the code_converter function and add them here!
  """
  class Dense:
    name = 'keras.layers.Dense'
    activations = [Activations.ReLU.name, Activations.Sigmoid.name, Activations.Softmax.name, Activations.Tanh.name, Activations.Linear.name]
    string = 'to_add'

  class Dropout:
    name = 'keras.layers.Dropout'
    activations = [None]
    string = None

  class SimpleRNN:
    name = 'keras.layers.SimpleRNN'
    activations = [Activations.Tanh]
    string = 'def simplernn(x, idx):\n' \
             '\tstates = [np.zeros(w[idx][0].shape[1], dtype=np.float32)]\n' \
             '\tfor step in range(x.shape[0]):\n' \
             '\t\tstates.append(np.tanh(np.dot(x[step], w[idx][0]) + np.dot(states[-1], w[idx][1]) + b[idx]))\n' \
             '\treturn np.array(states[1:])'


class SupportedAttrs:
  models = [Models.Sequential.name]

  layers = [Layers.Dense.name, Layers.Dropout.name, Layers.SimpleRNN.name]

  activations = [Activations.ReLU.name,
                 Activations.Sigmoid.name,
                 Activations.Softmax.name,
                 Activations.Tanh.name,
                 Activations.Linear.name]

  layers_without_activations = [Layers.Dropout.name]
  recurrent_layers = [Layers.SimpleRNN.name]
  ignored_layers = [Layers.Dropout.name]


class LayerInfo:
  supported = False
  has_activation = False
  returns_sequences = False
  is_recurrent = False
  is_ignored = False

  name = None
  activation = None
  weights = None
  biases = None


def code_converter(indentation_spaces=2):
  """
  :param indentation_spaces: Enter the number of spaces for each 'indent'
  :return: A string representation of the input function
  """
  print('This converts code into a supported string format for Konverter.')
  print('Simply paste your function here, with no extra indentation. Each indent will be replaced with a \\t character.')
  code = input()
  return code.replace(' ' * indentation_spaces, '\t')


watermark = '"""\n\tGenerated using Konverter: https://github.com/ShaneSmiskol/Konverter\n"""\n\n'
