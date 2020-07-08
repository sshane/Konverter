# Konverter ![Konverter Tests](https://github.com/ShaneSmiskol/Konverter/workflows/Konverter%20Tests/badge.svg)
### Convert your Keras models into pure Python 🐍+ NumPy.

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
  - BatchNormalization
    - Works with all supported layers
- Activations:
  - ReLU
  - Sigmoid
  - Softmax
  - Tanh
  - Linear/None

#### Roadmap 🛣
The project to do list can be [found here](https://github.com/ShaneSmiskol/Konverter/projects/1).

## Features 🤖
- Super quick conversion of your models. Takes less than a second. 🐱‍👤
- Usually reduces the size of Keras models by about 69.37%. 👌
- In some cases, prediction is quicker than Keras or SNPE (dense models). 🏎
  - RNNs: Since we lose the GPU using NumPy, predictions may be slower
- Stores the weights and biases of your model in a separate compressed NumPy file. 👇

## Benchmarks 📈
Benchmarks can be found in [BENCHMARKS.md](BENCHMARKS.md).

## Installation & Usage 🌍
### Install Konverter using pip:
`pip install keras-konverter`

### Konverting using the CLI: 🎹
`konverter ./My Models/my_model.h5 ./My Models/my_model.py"`  (py suffix is optional)

Type `konverter` to get all possible arguments and flags!

Arguments:
- `input_model`: Either the the location of your tf.keras .h5 model, or a preloaded Sequential model if using with Python. This is required
- `output_file`: Optional file path for your output model, along with the weights file. Default is same name, same directory
- Flags:
  - `--indent, -i`: How many spaces to use for indentation, default is 2
  - `--silent, -i`: Whether you want Konverter to silently Konvert
  - `--no-watermark, -nw`: Removes the watermark prepended to the output model file

### Konverting programmatically: 🤖
All parameters with defaults: `konverter.konvert(input_model, output_file=None, indent=2, silent=False, no_watermark=False, tf_verbose=False)`
```python
>>> import konverter
>>> konverter.konvert('examples/all_dense.h5', output_file='examples/all_dense.py')
```

*Note: The model file will be saved as `f'{output_file}.py'` and the weights will be saved as `f'{output_file}_weights.npz'` in the same directory.* ***Make sure to change the path inside the model wrapper if you move the files after Konversion.***

That's it! If your model is supported (check [Supported Keras Model Attributes](#Supported-Keras-Model-Attributes)), then your newly converted Konverter model should be ready to go.

To predict: Import your model wrapper and run the `predict()` function. ‼**Always double check that the outputs closely match your Keras model's**‼. Automatic verification will come soon. **For safety, always make sure your input is a `np.float32` array.**

```python
import numpy as np
from examples.all_dense import predict
predict(np.random.rand(200).astype(np.float32))
```

[See limitations and issues.](#Current-Limitations-and-Issues)

## Demo
<img src="https://raw.githubusercontent.com/ShaneSmiskol/Konverter/master/.media/konverter.gif?raw=true" width="913">


## Dependencies
Thanks to [@apiad](https://github.com/apiad) you can now use [Poetry](https://github.com/python-poetry/poetry) to install all the needed dependencies for this tool! However the requirements are a pretty short list:
- It seems most versions of TensorFlow that include Keras work perfectly fine. Tested from 1.14 to 2.1.0 using Actions and no issues have occurred.
  - **Important**: You must create your models with tf.keras currently (not keras)
- Python >= 3.6 (for the glorious f-strings!)

To install all needed dependencies, simply `cd` into the base directory of Konverter, and run:

```
poetry install --no-dev
```

If you would like to use this version of Konverter (not from pip), then you may need to also run `poetry shell` after to enter poetry's virtualenv environment. **If you go down this path, make sure to remove `--no-dev` so TensorFlow installs in the venv!**

## Current Limitations and Issues 😬
- Dimensionality of input data:

  When working with models using `softmax`, the dimensionality of the input data matters. For example, predicting on the same data with different input dimensionality sometimes results in different outputs:
  ```python
  >>> model.predict([[1, 3, 5]])  # keras model, correct output
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
- Batch prediction with SimpleRNN (and possibly all RNN) layers

  Currently, the converted model has no way of determining if you're feeding a single prediction or a batch of predictions, and it will fail to give the correct output in certain cases (more likely with recurrent layers and softmax dense outputs layers). Support will be added soon.
