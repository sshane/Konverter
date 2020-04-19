# Konverter
### A tool to convert Keras models to a single file using only NumPy.

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
    4. Linear

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

**Important (current limitations): The dimensionality of the input data matters when predicting with your Konverted model in certain cases. For example, with a `softmax`/`binary_crossentropy` model, executing the following will output all 1's:**
```python
>>> predict([[0.5]])
[[1.0, 1.0, 1.0, 1.0]]
```

**While the following will output the correct prediction:**
```python
>>> predict([0.5])
array([9.99955335e-01, 4.46511006e-05, 1.41283159e-08, 1.75172056e-17])
```

**And if trying a batch prediction with classes and `softmax`, the model fails completely:**
```python
>>> predict([[0.5], [0.5]])
array([[0.5, 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, 0.5]])
```

**Always double check that predictions are working correctly before deploying the model.**

Benchmark:
---
(see exact model in [build_test_model.py](https://github.com/ShaneSmiskol/Konverter/blob/0150ae6f22404521c9ff77f36a0047d7a95cbeb8/build_test_model.py)):
```
    samples: 10000

    Keras model batch prediction time: 0.451137s
    Konverted model batch prediction time: 0.298853s
    -----
    Keras model single prediction time: 149.628608s
    Konverted model single prediction time: 3.352782s
    
    keras vs. konverted model (comparing models, lower is better):
    Mean absolute error for 10000 predictions: 9.013858662641816e-07
    Mean squared error for 10000 predictions: 3.054752549391908e-12
```

Benchmark info:
---
The batch predictions are simply that, 10,000 random samples are fed into each model to be predicted on all at once. This is usually the fastest method of executing a prediction for a lot of unrelated samples.

With the single predictions, we are predicting on the same samples as before, however we are using a loop and predicting on each sample one by one. This is usually how you will be executing predictions in production. You won't know future data, so this is a good way to benchmark inference times for both model formats.

The errors at the end of the benchmark are the mean absolute error (`np.mean(np.abs(keras_preds - konverter_preds))`) and mean squared error (`np.mean((keras_preds - konverter_preds) ** 2)`) which are two common methods of measuring prediction vs. ground truth error. 3.05e-12 for MSE essentially means the two model formats are predicting the same outputs up to an accuracy of ~12 decimal places (correct me if I'm wrong) over 10,000 predictions.