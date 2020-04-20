class Aliases:
  class models:
    sequential = 'sequential'

  class layers:
    dense = 'keras.layers.Dense'
    dropout = 'keras.layers.Dropout'

  class activations:
    relu = 'keras.activations.relu'
    sigmoid = 'keras.activations.sigmoid'
    softmax = 'keras.activations.softmax'
    tanh = 'keras.activations.tanh'
    linear = 'keras.activations.linear'


class SupportedAttrs:
  models = ['sequential']

  layers = [Aliases.layers.dense, Aliases.layers.dropout]

  activations = [Aliases.activations.relu,
                 Aliases.activations.sigmoid,
                 Aliases.activations.softmax,
                 Aliases.activations.tanh,
                 Aliases.activations.linear]

  layers_without_activations = [Aliases.layers.dropout]


class LayerInfo:
  supported = False
  has_activation = False
  name = None
  activation = None
  weights = None
  biases = None


class AttrStrings:
  activations = {Aliases.activations.relu: 'def relu(x):\nreturn np.maximum(0, x)',
                 Aliases.activations.sigmoid: 'def sigmoid(x):\nreturn 1 / (1 + np.exp(-x))',
                 Aliases.activations.softmax: 'def softmax(x):\nreturn np.exp(x) / np.sum(np.exp(x), axis=0)',
                 Aliases.activations.tanh: 'def tanh(x):\nreturn np.tanh(x)'}
