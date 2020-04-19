# Konverter
#### A tool to convert Keras models to a single file using only NumPy.

---
The goal of this tool is to provide a quick and easy way to execute simple Keras models on machines or setups where utilizing TensorFlow/Keras is impossible. Specifically, in my case, to replace SNPE (Snapdragon Neural Processing Engine) for inference on phones with Python.

---
This is a work in progress, currently only the following are supported:

  - Models:
    1. Sequential
  - Layers:
    1. Dense
  - Activations:
    1. ReLU
    2. Linear

The following are guaranteed to work (2D data might work but is untested at this early stage):

  - Input data shape:
    1. 1-dimensional input samples, eg. `x_train = np.array([[1], [2], [3]])`
  - Output data shape:
    1. 1-dimensional output samples, eg. `y_train = np.array([1, 2, 3])`

Benchmark (see exact model in [build_test_model.py](build_test_model.py)):
```
    Keras model batch prediction time: 0.405091s
    Konverted model batch prediction time: 0.005002s
    -----
    Keras model single prediction time: 155.646902s
    Konverted model single prediction time: 0.119027s
    
    keras vs. konverted model:
    Mean absolute error for 10000 predictions: 2.434262684936961e-07
    Mean squared error for 10000 predictions: 9.782740443465008e-14
```