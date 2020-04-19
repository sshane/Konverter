class SupportedAttrs:
  activations = {'keras.activations.relu': 'relu',
                 'keras.activations.sigmoid': 'sigmoid',
                 'keras.activations.softmax': 'softmax',
                 'keras.activations.linear': 'linear'}
  layers = {'keras.layers.Dense': 'dense'}
  models = ['sequential']


class LayerInfo:
  supported = False
  name = None
  activation = None
  weights = None
  biases = None


class ActivationFunctions:
  activations = {'relu': 'def relu(x):\nreturn np.maximum(0, x)',
                 'sigmoid': 'def sigmoid(x):\nreturn 1 / (1 + np.exp(-x))',
                 'softmax': 'def softmax(x):\nreturn np.exp(x) / np.sum(np.exp(x), axis=0)'}
