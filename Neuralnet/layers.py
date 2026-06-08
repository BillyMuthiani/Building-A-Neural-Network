

import numpy as np


class Dense:
    def __init__(self, input_size, output_size):

        self.weights = np.random.randn(
            input_size,
            output_size
        ) * 0.01

        self.biases = np.zeros(
            (1, output_size)
        )

    def forward(self, X):

        self.input = X

        return np.dot(
            X,
            self.weights
        ) + self.biases