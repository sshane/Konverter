# Konverter Benchmarks

## Snapdragon 821 (LeEco Le Pro3) - 10,000 random single predictions
Comparison of a model converted with SNPE 1.19 (Snapdragon Neural Processing Engine) and the same model converted with Konverter.

|              |   SNPE model   | Konverted model |
| ------------ | -------------- | --------------- |
| Total time   | 16.150222 sec. | 10.021809 sec.  |
| Average time | 0.0016150 sec. | 0.0010022 sec.  |
| Model rate   | 619.18654 Hz   | 997.82385 Hz    |


The model:

```python
model = Sequential()
model.add(Dense(204, activation='relu', input_shape=(103,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
```

---
## Ryzen 5 3600 (Desktop) - 10,000 random predictions
(see exact model in [build_test_model.py](https://github.com/ShaneSmiskol/Konverter/blob/0150ae6f22404521c9ff77f36a0047d7a95cbeb8/build_test_model.py)):

### Batch prediction:

|              | Keras model    | Konverted model |
| ------------ | -------------- | --------------- |
| Total time   | 0.403091 sec.  | 0.088019 sec.   |

### Single prediction:

|              | Keras model     | Konverted model |
| ------------ | --------------- | --------------- |
| Total time   | 135.074061 sec. | 1.848414 sec.   |
| Average time | 0.01350741 sec. | 0.000185 sec.   |
| Model rate   | 74.0334593 Hz   | 5410.043 Hz     |

---
## Benchmark info:

The batch predictions are simply that, 10,000 random samples are fed into each model to be predicted on all at once. This is usually the fastest method of executing a prediction for a lot of unrelated samples.

With the single predictions, we are predicting on the same samples as before, however we are using a loop and predicting on each sample one by one. This is usually how you will be executing predictions in production. You won't know future data, so this is a good way to benchmark inference times for both model formats.
