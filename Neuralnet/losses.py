import numpy as np


class MSE:

    def forward(self, y_true, y_pred):

        return np.mean(
            (y_true - y_pred) ** 2
        )

    def backward(self, y_true, y_pred):

        samples = len(y_true)

        return (
            -2
            * (y_true - y_pred)
            / samples
        )


class BinaryCrossEntropy:

    def forward(
        self,
        y_true,
        y_pred
    ):

        y_pred = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        loss = -np.mean(
            y_true * np.log(y_pred)
            +
            (1 - y_true)
            * np.log(
                1 - y_pred
            )
        )

        return loss

    def backward(
        self,
        y_true,
        y_pred
    ):

        y_pred = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        samples = len(y_true)

        return (
            -(y_true / y_pred)
            +
            ((1 - y_true)
             / (1 - y_pred))
        ) / samples
    
class CategoricalCrossEntropy:

    def forward(
        self,
        y_true,
        y_pred
    ):

        samples = len(y_pred)

        y_pred = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        correct_confidences = (
            y_pred[
                range(samples),
                y_true
            ]
        )

        return -np.mean(
            np.log(
                correct_confidences
            )
        )

    def backward(
        self,
        y_true,
        y_pred
    ):

        samples = len(y_pred)

        labels = len(
            y_pred[0]
        )

        y_true = np.eye(
            labels
        )[y_true]

        return (
            -y_true
            / y_pred
        ) / samples
