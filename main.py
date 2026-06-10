import numpy as np

from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU
from Neuralnet.activations import Sigmoid
from Neuralnet.activations import Tanh

from Neuralnet.losses import BinaryCrossEntropy
from Neuralnet.metrics import Accuracy
from Neuralnet.optimizers import SGD

from Neuralnet.model import Sequential


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


model = Sequential()

model.add(Dense(2,8))
model.add(Tanh())

model.add(Dense(8,1))
model.add(Sigmoid())


loss = BinaryCrossEntropy()
metric = Accuracy()

optimizer = SGD(
    learning_rate=0.1
)

model.fit(
    X,
    y,
    loss,
    optimizer,
    epochs=20000,
    metric=metric
)

predictions = model.predict(X)

print(predictions)