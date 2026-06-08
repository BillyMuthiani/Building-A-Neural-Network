# main.py

import numpy as np

from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU
from Neuralnet.model import Sequential


model = Sequential()

model.add(Dense(2, 4))
model.add(ReLU())
model.add(Dense(4, 1))

X = np.array([
    [1, 0],
    [0, 1]
])

output = model.forward(X)

print(output)