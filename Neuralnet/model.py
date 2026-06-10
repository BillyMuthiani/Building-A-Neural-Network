class Sequential:

    def __init__(self):

        self.layers = []

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, X):

        output = X

        for layer in self.layers:

            output = layer.forward(output)

        return output

    def backward(self, dvalues):

        for layer in reversed(self.layers):

            dvalues = layer.backward(dvalues)

    def predict(self, X):

        return self.forward(X)