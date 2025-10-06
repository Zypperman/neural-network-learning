import numpy as np
from util import NLL


class CustomPerception:
    def __init__(self, num_of_weights: int) -> None:
        """
        initialize weights.

        Args:
            num_of_weights (_type_): _description_
            bias (_type_): _description_
        """
        self.num_of_weights = num_of_weights
        self.bias = np.random.randn()
        self.weights_arr = np.random.rand(self.num_of_weights)

    def forward_pass(self, inputs):
        if len(inputs) != self.num_of_weights:
            raise AttributeError("wrong number of inputs")
        return np.dot(inputs, self.weights_arr) + self.bias

    def update_weights(self, new_weights: list) -> int:
        try:
            self.weights = new_weights
            return 0
        except Exception:
            raise AttributeError("unable to update")

class Optimizer:
    def __init__(self, perceptron, loss="NLL", epochs=1):
        """
        optimizer for 1 epoch LD

        Args:
            perceptron (perceptron): _description_
            loss (str, optional): _description_. Defaults to "NLL".
            epochs (int, optional): _description_. Defaults to 1.
        """
        self.perceptron = perceptron
        if loss == "NLL":
            self.loss = NLL
        else:
            self.loss = None




