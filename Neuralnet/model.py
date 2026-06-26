class Sequential:

    def __init__(self):

        self.layers = []
        self.optimizer = None
        self.loss_function = None
        self.metric = None

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

    def compile(
        self,
        loss,
        optimizer,
        metric=None
    ):

        self.loss_function = loss
        self.optimizer = optimizer
        self.metric = metric

    def fit(
        self,
        X,
        y,
        epochs=1000
    ):

        if self.loss_function is None:
            raise ValueError(
                "Compile the model before calling fit()."
            )

        history = {
            "loss": []
        }

        if self.metric:
            history["accuracy"] = []

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss = self.loss_function.forward(
                y,
                predictions
            )

            history["loss"].append(loss)

            dloss = self.loss_function.backward(
                y,
                predictions
            )

            self.backward(dloss)

            for layer in self.layers:
                self.optimizer.update(layer)

            if self.metric:

                score = self.metric.calculate(
                    y,
                    predictions
                )

                history["accuracy"].append(score)

                if epoch % 100 == 0:
                    print(
                        f"Epoch {epoch} "
                        f"Loss: {loss:.6f} "
                        f"Accuracy: {score:.4f}"
                    )

            else:

                if epoch % 100 == 0:
                    print(
                        f"Epoch {epoch} "
                        f"Loss: {loss:.6f}"
                    )

        return history