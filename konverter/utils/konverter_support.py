from konverter.utils.model_attributes import BaseLayerInfo, BaseModelInfo, Models, Activations, Layers
import numpy as np


class KonverterSupport:
  def __init__(self):
    self.models = [getattr(Models, i) for i in dir(Models) if '_' not in i]  # classes, not names
    self.layers = [getattr(Layers, i) for i in dir(Layers) if '_' not in i]
    self.activations = [getattr(Activations, i) for i in dir(Activations) if '_' not in i]

    self.attrs_without_activations = [Layers.Dropout.name, Activations.Linear.name]
    self.recurrent_layers = [Layers.SimpleRNN.name, Layers.GRU.name]
    self.ignored_layers = [Layers.Dropout.name]

  def get_class_from_name(self, name, search_in):
    """
    :param name: A name of an attribute, ex. keras.layers.Dense, keras.activations.relu
    :param search_in: A class list to search, ex. 'layers', 'models'
    :return: A class object of the attribute name, or False if not found/supported
    """
    attrs = getattr(self, search_in, None)
    for attr_class in attrs:
      if name == attr_class.name:
        return attr_class()  # new instance of class
    return False

  def in_models(self, name):
    return name in [mdl.name for mdl in self.models]

  def in_layers(self, name):
    return name in [lyr.name for lyr in self.layers]

  def in_activations(self, name):
    return name in [act.name for act in self.activations]

  def model_activations(self, ls):
    """
    :param ls: layers
    :return: A set of all the activations used in the layers
    """
    a = []
    for lyr in ls:
      if lyr.info.has_activation:
        a.append(lyr.info.activation.name)
    return set(a)

  def attr_map(self, classes, attr):
    """Takes a list of (layer/activation/model) classes and returns the specified attribute from each"""
    return list(map(lambda cls: getattr(cls, attr), classes))

  def get_model_info(self, model):
    name = getattr(model, '_keras_api_names_v1')[0]
    model_class = self.get_class_from_name(name, 'models')
    if not model_class:
      model_class = Models.Unsupported()
      model_class.name = name
      return model_class

    model_class.info = BaseModelInfo()
    model_class.info.input_shape = model.input_shape
    model_class.info.supported = True
    return model_class

  def get_layer_info(self, layer):
    name = getattr(layer, '_keras_api_names_v1')
    if not len(name):
      name = getattr(layer, '_keras_api_names')
    layer_class = self.get_class_from_name(name[0], 'layers')  # assume only one name
    if not layer_class:
      layer_class = Layers.Unsupported()  # add activation below to raise exception with
      layer_class.name = name

    layer_class.info = BaseLayerInfo()
    layer_class.info.is_ignored = layer_class.name in self.ignored_layers

    is_linear = False
    if layer_class.name not in self.attrs_without_activations:
      activation = getattr(layer.activation, '_keras_api_names')
      if len(activation) == 1:
        layer_class.info.activation = self.get_class_from_name(activation[0], 'activations')
        if layer_class.info.activation.name not in self.attrs_without_activations:
          layer_class.info.has_activation = True
        else:
          is_linear = True
      else:
        raise Exception('None or multiple activations?')

    if layer_class.info.has_activation:
      # check layer activation against this layer's supported activations
      if layer_class.info.activation.name in self.attr_map(layer_class.supported_activations, 'name'):
        layer_class.info.supported = True
    elif layer_class.info.is_ignored or is_linear:  # skip activation check if layer has no activation (eg. dropout or linear)
      layer_class.info.supported = True

    if not layer_class.info.supported or (not is_linear and not layer_class.info.has_activation):
      return layer_class

    wb = layer.get_weights()
    if len(wb) == 2:
      weights, biases = wb
    elif len(wb) == 3 and layer_class.name in self.recurrent_layers:
      *weights, biases = layer.get_weights()
      layer_class.info.returns_sequences = layer.return_sequences
      layer_class.info.is_recurrent = True
    else:
      raise Exception('Layer `{}` had an unsupported number of weights: {}'.format(layer_class.name, len(wb)))

    layer_class.info.weights = np.array(weights)
    layer_class.info.biases = np.array(biases)
    return layer_class
