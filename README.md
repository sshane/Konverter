# Konverter
### Convert your (simple) Keras models into pure Python + NumPy.

The goal of this tool is to provide a quick and easy way to execute simple Keras models on machines or setups where utilizing TensorFlow/Keras is impossible. Specifically, in my case, to replace SNPE (Snapdragon Neural Processing Engine) for inference on phones with Python.

---
This tool is a work in progress, currently only the following are supported:

  - Models:
    1. Sequential
  - Layers:
    1. Dense
  - Activations:
    1. ReLU
    2. Sigmoid
    3. Softmax
    4. Tanh
    5. Linear/None

The following data shapes are guaranteed to work (pretty much anything you can do with dense):

  - Input data shape:
    1. 1-dimensional samples, eg. `x_train = np.array([[1], [2], [3]])`
    2. 2-dimensional samples, eg. `x_train = np.array([[1, 2], [2, 3], [3, 4]])`
  - Output data shape:
    1. 1-dimensional samples, eg. `y_train = np.array([1, 2, 3])`
    2. 2-dimensional samples, eg. `y_train = np.array([[1, 2], [3, 4], [5, 6]])`

Usage:
---
To be added.

**Important (current limitations): When working with models using `softmax`, the dimensionality of the input data matters. For example, with certain models (specifically `softmax`), predicting on the same data with an incorrect dimensionality could result in an incorrect prediction:**
```python
>>> model.predict([[1, 3, 5]])  # keras model
array([[14.792273, 15.59787 , 15.543163]])
>>> predict([[1, 3, 5]])  # Konverted model, wrong dimensionality
array([[11.97839948, 18.09931636, 15.48014805]])
>>> predict([1, 3, 5])  # And correct dimensionality
array([14.79227209, 15.59786987, 15.54316282])
```

**If trying a batch prediction with classes and `softmax`, the model fails completely:**
```python
>>> predict([[0.5], [0.5]])
array([[0.5, 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, 0.5]])
```

**Always double check that predictions are working correctly before deploying the model.**

