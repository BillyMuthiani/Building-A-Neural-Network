import numpy as np


class ReLU:

    def forward(self, X):

        self.input = X

        return np.maximum(
            0,
            X
        )
    
class Sigmoid:

    def forward(self, X):

        self.input = X

        return 1 / (
            1 + np.exp(-X)
        )    