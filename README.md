# Konverter ![Konverter Tests](https://github.com/ShaneSmiskol/Konverter/workflows/Konverter%20Tests/badge.svg)
### Convert your Keras models into pure Python + NumPy.

The goal of this tool is to provide a quick and easy way to execute Keras models on machines or setups where utilizing TensorFlow/Keras is impossible. Specifically, in my case, to replace SNPE (Snapdragon Neural Processing Engine) for inference on phones with Python.

## Supported Keras Model Attributes
- Models:
  - Sequential
- Layers:
  - Dense
  - Dropout
    - Will be ignored during inference (SNPE 1.19 does NOT support dropout with Keras!)
  - SimpleRNN
    - Batch predictions do not currently work correctly.
  - GRU
    - **Important:** The current GRU support is based on [`GRU v3`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU) in tf.keras 2.1.0. It will not work correctly with older versions of TensorFlow.
    - Batch prediction untested 
- Activations:
  - ReLU
  - Sigmoid
  - Softmax
  - Tanh
  - Linear/None

#### Roadmap
The project to do list can be [found here](https://github.com/ShaneSmiskol/Konverter/projects/1).

## Features
- Super quick conversion of your models. Takes less than a second.
- Usually reduces the size of Keras models by about 69.37%.
- In some cases, prediction is quicker than Keras or SNPE (dense models).
  - RNNs: Since we lose the GPU using NumPy, predictions may be slower
- Stores the weights and biases of your model in a separate compressed NumPy file.

## Benchmarks
Benchmarks can be found in [BENCHMARKS.md](BENCHMARKS.md).

## Usage
*To update.*

~~To convert your Keras model, simply edit the last few lines in [konverter.py](konverter.py#L175).~~

~~1. For the `model` variable, you'll want to replace the path with the location of your Keras `.h5` model.
2. For the `output_file` variable, enter your desired output model name. The model file will be saved as `f'{}.py'` and the weights will be saved as `f'{}_weights.npz'` in the same directory.
3. Finally, enter the number of spaces to use as indentation and run with `python konverter.py`!~~

That's it! If your model is supported (check [Supported Keras Model Attributes](#Supported-Keras-Model-Attributes)), then your newly converted Konverter model should be ready to go.

To predict: Run `predict()` function in your Python model. Always double check that the outputs closely match your Keras model's.

Nesting your input data with the wrong number of arrays/lists can sometimes cause the outputs to be complete incorrect; you may need to experiment with `predict[[sample]])` vs. `predict([sample])` for example.

## Demo:
<img src=".media/konverter.gif?raw=true" width="913">


## Dependencies
Thanks to [@apiad](https://github.com/apiad) you can now use [Poetry](https://github.com/python-poetry/poetry) to install all the needed dependencies for this tool! However the requirements are a pretty short list:
- It seems most versions of TensorFlow that include Keras work perfectly fine. Tested from 1.14 to 2.1.0 using Actions and no issues have occurred.
  - **Important**: You must create your models with tf.keras currently (not keras)
- Python >= 3.6 (for the glorious f-strings!)
- [Typer](https://github.com/tiangolo/typer/issues), requires >= 3.6

To install all needed dependencies, simply `cd` into the base directory of Konverter, and run:

```
poetry install --no-dev
```

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
- Batch prediction with SimpleRNN layers

  Currently, the converted model has no way of determining if you're feeding a single prediction or a batch of predictions, and it will fail to give the correct output. Support will be added soon.
