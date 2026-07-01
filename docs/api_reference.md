# API Reference

Complete reference for all exported symbols in Kronyx.

## Layers

### Dense

Fully connected layer.

```python
from kronyx import Dense

layer = Dense(input_size, output_size, initializer="he_normal", kernel_regularizer=None)
```

**Parameters:**
- `input_size` (int): Number of input features
- `output_size` (int): Number of output neurons
- `initializer` (str): Weight initialization method
- `kernel_regularizer` (L2): Optional weight penalty

### Conv2D

2D convolutional layer.

```python
from kronyx import Conv2D

layer = Conv2D(filters, kernel_size, stride=1, padding="valid")
```

### Flatten

Flattens multi-dimensional input to 1D.

```python
from kronyx import Flatten

layer = Flatten()
```

### Dropout

Regularization layer.

```python
from kronyx import Dropout

layer = Dropout(rate=0.5)
```

### BatchNormalization

Normalizes layer inputs.

```python
from kronyx import BatchNormalization

layer = BatchNormalization(momentum=0.9, epsilon=1e-5)
```

## Activations

### ReLU

Rectified Linear Unit activation.

```python
from kronyx import ReLU

output = ReLU().forward(x)
```

### Sigmoid

Sigmoid activation.

```python
from kronyx import Sigmoid

output = Sigmoid().forward(x)
```

### Tanh

Hyperbolic tangent activation.

```python
from kronyx import Tanh

output = Tanh().forward(x)
```

### Softmax

Softmax activation for multi-class output.

```python
from kronyx import Softmax

output = Softmax().forward(x)
```

## Models

### Sequential

Sequential neural network container.

```python
from kronyx import Sequential

model = Sequential()
model.add(Dense(10, 32))
model.compile(loss=..., optimizer=..., metric=...)
model.fit(X, y, epochs=100)
```

**Methods:**
- `add(layer)`: Add a layer
- `compile(loss, optimizer, metric)`: Configure training
- `fit(X, y, epochs, ...)`: Train the model
- `predict(X, ...)`: Make predictions
- `evaluate(X, y, ...)`: Evaluate on data
- `save(filename)`: Save to .krx
- `save_weights(filename)`: Save weights to .npz
- `load_weights(filename)`: Load weights
- `to_json()`: Export architecture
- `from_json(json_string)`: Create from JSON
- `summary()`: Print model summary

### History

Training history container.

```python
history = model.fit(X, y)
history.summary()  # Print metrics summary
```

## Optimizers

### SGD

Stochastic Gradient Descent optimizer.

```python
from kronyx import SGD

optimizer = SGD(learning_rate=0.01)
```

### Adam

Adam optimizer with momentum.

```python
from kronyx import Adam

optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
```

## Losses

### BinaryCrossEntropy

Binary cross-entropy loss.

```python
from kronyx import BinaryCrossEntropy

loss = BinaryCrossEntropy()
```

### CategoricalCrossEntropy

Categorical cross-entropy loss.

```python
from kronyx import CategoricalCrossEntropy

loss = CategoricalCrossEntropy()
```

### SoftmaxCategoricalCrossEntropy

Combined softmax and cross-entropy loss.

```python
from kronyx import SoftmaxCategoricalCrossEntropy

loss = SoftmaxCategoricalCrossEntropy()
```

## Metrics

### Accuracy

Classification accuracy metric.

```python
from kronyx import Accuracy

metric = Accuracy()
```

## Regularizers

### L2

L2 weight regularization.

```python
from kronyx import L2

reg = L2(lambda_=0.01)
```

## Serialization

### load_model

Load a model from .krx file.

```python
from kronyx import load_model

model = load_model('model.krx')
```

### from_json

Create model from JSON architecture.

```python
from kronyx import from_json

model = Sequential.from_json(json_string)
```

### SerializationError

Exception raised for serialization failures.

```python
from kronyx import SerializationError

try:
    model = load_model('missing.krx')
except SerializationError as e:
    print(f"Failed to load: {e}")
```

## Callbacks

### EarlyStopping

Stop training early if metric stops improving.

```python
from kronyx import EarlyStopping

callback = EarlyStopping(patience=10, min_delta=0.001)
```

### ModelCheckpoint

Save model after each epoch.

```python
from kronyx import ModelCheckpoint

callback = ModelCheckpoint(filepath='model.krx', save_best_only=True)
```

### CSVLogger

Log training history to CSV.

```python
from kronyx import CSVLogger

callback = CSVLogger(filename='log.csv')
```

### ReduceLROnPlateau

Reduce learning rate on plateau.

```python
from kronyx import ReduceLROnPlateau

callback = ReduceLROnPlateau(factor=0.5, patience=5)
```

## Utilities

### set_seed

Set random seed for reproducibility.

```python
from kronyx import set_seed

set_seed(42)
```

### Initializers

```python
from kronyx import he_normal, xavier_uniform, lecun_normal

weights = he_normal(fan_in, fan_out)
```

## Exceptions

- `NeuralnetError`: Base exception class
- `ConfigurationError`: Configuration errors
- `NotCompiledError`: Model not compiled
- `ShapeError`: Shape mismatch errors
- `OptimizerError`: Optimizer errors
- `SerializationError`: Serialization errors