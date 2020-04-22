class Models:
  class Sequential:
    name = 'sequential'


class Activations:
  """
    The class that contains the supported activations and any information we will need to generate models
      ex. activation in string format
    To add new activations, use the code_converter function and add them here!
  """
  class BaseAcivation:
    name = None
    alias = None
    string = None
    needs_function = True

  class ReLU(BaseAcivation):
    name = 'keras.activations.relu'
    alias = 'relu'
    string = 'np.maximum(0, {})'
    needs_function = False

  class Sigmoid(BaseAcivation):
    name = 'keras.activations.sigmoid'
    alias = 'sigmoid'
    string = 'def sigmoid(x):\n\treturn 1 / (1 + np.exp(-x))'

  class Softmax(BaseAcivation):
    name = 'keras.activations.softmax'
    alias = 'softmax'
    string = 'def softmax(x):\n\treturn np.exp(x) / np.sum(np.exp(x), axis=0)'

  class Tanh(BaseAcivation):
    name = 'keras.activations.tanh'
    alias = 'tanh'
    string = 'np.tanh({})'  # don't define a function if you don't want your string added to file as a function
    needs_function = False


  class Linear(BaseAcivation):
    name = 'keras.activations.linear'
    alias = 'linear'


class Layers:
  """
    The class that contains the supported layers and any information we will need to generate models
      ex. function in string format
    To add new layers, use the code_converter function and add them here!
  """
  class BaseLayer:
    name = None
    alias = None
    activations = [None]
    string = None

  class Dense(BaseLayer):
    name = 'keras.layers.Dense'
    alias = 'dense'
    activations = [Activations.ReLU.name, Activations.Sigmoid.name, Activations.Softmax.name, Activations.Tanh.name, Activations.Linear.name]
    string = 'np.dot({}, w[{}]) + b[{}]'  # n0 is the previous layer, n1 is weight, n2 is bias

  class Dropout(BaseLayer):
    name = 'keras.layers.Dropout'
    alias = 'dropout'

  class SimpleRNN(BaseLayer):
    name = 'keras.layers.SimpleRNN'
    alias = 'SimpleRNN'
    activations = [Activations.Tanh.name]
    string = 'def simplernn(x, idx):\n' \
             '\tstates = [np.zeros(w[idx][0].shape[1], dtype=np.float32)]\n' \
             '\tfor step in range(x.shape[0]):\n' \
             '\t\tstates.append(np.tanh(np.dot(x[step], w[idx][0]) + np.dot(states[-1], w[idx][1]) + b[idx]))\n' \
             '\treturn np.array(states[1:])'

  class Unsupported(BaseLayer):  # propogated with layer info and returned to Konverter if layer is unsupported
    pass


class BaseLayerInfo:
  supported = False
  has_activation = False
  returns_sequences = False
  is_recurrent = False
  is_ignored = False

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
