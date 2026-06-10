import numpy as np


class ReLU:

    def forward(self, X):

        self.input = X

        return np.maximum(
            0,
            X
        )

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        self.dinputs[
            self.input <= 0
        ] = 0

        return self.dinputs


    
class Sigmoid:

    def forward(self, X):

        self.output = 1 / (1 + np.exp(-X))

        return self.output

    def backward(self, dvalues):

        self.dinputs = (
            dvalues
            * self.output
            * (1 - self.output)
        )

        return self.dinputs
    
class Tanh:

    def forward(self, X):

        self.output = np.tanh(X)

        return self.output

    def backward(self, dvalues):

        return (
            dvalues *
            (1 - self.output**2)
        )