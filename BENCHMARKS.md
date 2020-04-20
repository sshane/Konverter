# Konverter
### Benchmarks

Comparing a model converted with SNPE (Snapdragon Neural Processing Engine) and the same model converted with Konverter, here are the results:

The model:

```python
model = Sequential()
model.add(Dense(204, activation='relu', input_shape=(103,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
```

### Snapdragon 821 (LeEco Le Pro3) - 10,000 random single predictions

|              |   SNPE model   | Konverted Model |
| ------------ | -------------- | --------------- |
| Total time   | 16.150222 sec. | 10.021809s sec. |
| Average time | 0.0016150 sec. | 0.0010022s sec. |
| Model rate   | 619.18654 Hz   | 997.82385 Hz    |

