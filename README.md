# Konverter (WIP)
### Convert your (simple) Keras models into pure Python + NumPy.

The goal of this tool is to provide a quick and easy way to execute simple Keras models on machines or setups where utilizing TensorFlow/Keras is impossible. Specifically, in my case, to replace SNPE (Snapdragon Neural Processing Engine) for inference on phones with Python.

## Supported Keras Model Attributes
- Models:
  - Sequential
- Layers:
  - Dense
  - Dropout
    - Will be ignored during inference (SNPE 1.19 does NOT support dropout with Keras!)
- Activations:
  - ReLU
  - Sigmoid
  - Softmax
  - Tanh
  - Linear/None
- Data shapes:
  - Pretty much anything you can do with dense layers, Konverter supports. 1D/2D input? Check. 1D/2D output? Check.

**Todo**:
- [ ] GRU
- [ ] LSTM
- [ ] Conv2D

## Features
- Super quick conversion of your models. Takes less than a second.
- Usually reduces the size of Keras models by about 69.37%.
- Prediction is usually quicker in most cases than Keras or SNPE.
- Stores the weights and biases of your model in a separate compressed NumPy file.
  - If your model output name is `dense_model`, the Python wrapper file will be named `dense_model.py` and the weights will be named `dense_model_weights.npz` in the same directory.

## Benchmarks
Benchmarks can be found in [BENCHMARKS.md](BENCHMARKS.md).

## Usage
To be added.

<img src="gifs/konverter.gif?raw=true" width="913">


## Requirements
I've built and tested Konverter with the following:
- TensorFlow 2.1.0
- tf.keras (not keras)
- Python >= 3.6 (for the glorious f-strings!)

## Current Limitations and Issues
- Dimensionality of input data:

  When working with models using `softmax`, the dimensionality of the input data matters. For example, predicting on the same data with different input dimensionality sometimes results in different outputs:
  ```python
  >>> model.predict([[1, 3, 5]])  # keras model
  array([[14.792273, 15.59787 , 15.543163]])
  >>> predict([[1, 3, 5]])  # Konverted model, wrong output
  array([[11.97839948, 18.09931636, 15.48014805]])
  >>> predict([1, 3, 5])  # And correct output
  array([14.79227209, 15.59786987, 15.54316282])
  ```

  If trying a batch prediction with classes and `softmax`, the model fails completely:
  ```python
  >>> predict([[0.5], [0.5]])
  array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
  ```

  Always double check that predictions are working correctly before deploying the model.
