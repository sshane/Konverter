class Aliases:
  class models:
    sequential = 'sequential'

  class layers:
    dense = 'keras.layers.Dense'
    dropout = 'keras.layers.Dropout'
    simplernn = 'keras.layers.SimpleRNN'

  class activations:
    relu = 'keras.activations.relu'
    sigmoid = 'keras.activations.sigmoid'
    softmax = 'keras.activations.softmax'
    tanh = 'keras.activations.tanh'
    linear = 'keras.activations.linear'


class SupportedAttrs:
  models = ['sequential']

  layers = [Aliases.layers.dense, Aliases.layers.dropout, Aliases.layers.simplernn]

  activations = [Aliases.activations.relu,
                 Aliases.activations.sigmoid,
                 Aliases.activations.softmax,
                 Aliases.activations.tanh,
                 Aliases.activations.linear]

  layers_without_activations = [Aliases.layers.dropout]
  recurrent_layers = [Aliases.layers.simplernn]
  ignored_layers = [Aliases.layers.dropout]


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


class AttrStrings:
  activations = {Aliases.activations.relu: 'def relu(x):\n\treturn np.maximum(0, x)',
                 Aliases.activations.sigmoid: 'def sigmoid(x):\n\treturn 1 / (1 + np.exp(-x))',
                 Aliases.activations.softmax: 'def softmax(x):\n\treturn np.exp(x) / np.sum(np.exp(x), axis=0)',
                 Aliases.activations.tanh: 'def tanh(x):\n\treturn np.tanh(x)'}

  layers = {Aliases.layers.simplernn: 'def simplernn(x, idx):\n\tstates = [np.zeros(w[idx][0].shape[1], dtype=np.float32)]\n\tfor step in range(x.shape[0]):\n\t\tstates.append(np.tanh(np.dot(x[step], w[idx][0]) + np.dot(states[-1], w[idx][1]) + b[idx]))\n\treturn np.array(states[1:])'}


watermark = '"""\n\tGenerated using Konverter: https://github.com/ShaneSmiskol/Konverter\n"""\n\n'
