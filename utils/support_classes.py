class SupportedAttrs:
  activations = {'keras.activations.relu': 'relu', 'keras.activations.linear': 'linear'}
  layers = {'keras.layers.Dense': 'dense', 'dense': 'dense'}
  models = ['sequential']


class LayerInfo:
  supported = False
  name = None
  activation = None
  weights = None
  biases = None


class ActivationFunctions:
  activations = {'relu': 'def relu(x):\nreturn np.maximum(0, x)'}
