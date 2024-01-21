import pickle
from enum import Enum, auto
from typing import Any, Optional, Self

import torch
import torch.nn.functional as F
from flamenn.layers import PerceptronLayer
from flamenn.networks import MultiLayerPerceptron

from ..types import Data, IBaseClassifier, Label, Path


class MLPParam(Enum):
    """MLP Parameters

    Members:
        INPUT_SIZE: The size of the input layer
        OUTPUT_SIZE: The size of the output layer
        LAYER_SIZES: The hidden layers (list of PerceptronLayer)
        LEARNING_RATE: The learning rate of the optimizer
    """

    INPUT_SIZE = auto()
    OUTPUT_SIZE = auto()
    LAYERS = auto()
    LEARNING_RATE = auto()
    CRITERION = auto()


class MLP(IBaseClassifier):
    def __init__(self, model: MultiLayerPerceptron) -> None:
        super().__init__()
        self.model: MultiLayerPerceptron = model

    @staticmethod
    def load(filename: Path, params: dict[MLPParam, Any] | None = None):
        """Load a pretrained MLP model

        Args:
            filename: Path to the model weights
            params: Parameters of the model
        """
        if params is None:
            raise NotImplementedError(
                "No default parameters for MLP have been decided yet"
            )

        mlp = MLP.new(params)

        with open(filename, "rb") as file:
            weights = pickle.load(file)

        mlp.model.load_state_dict(weights)

        return mlp

    @staticmethod
    def new(params: dict[MLPParam, Any]):
        """Constructor

        Args:
            params: Dictionary with model parameters

        Returns:
            The unfitted model with set parameters
        """
        model = MultiLayerPerceptron(input_size=int(params[MLPParam.INPUT_SIZE]))

        # Add the hidden layers
        for layer in params[MLPParam.LAYERS]:
            model.addLayer(layer)

        # Add the output layer
        model.addLayer(
            PerceptronLayer(int(params[MLPParam.OUTPUT_SIZE]), F.log_softmax, False)
        )
        model.addCriterion(params[MLPParam.CRITERION])
        model.addOptim("adam", learning_rate=params[MLPParam.LEARNING_RATE])

        return MLP(model)

    def save(self: Self, filename: Path) -> None:
        """Save a model"""
        raise NotImplementedError

    def predict(
        self: Self, X: Data, label_names: Optional[list[str]] = None
    ) -> list[dict[Label, float]]:
        """Predict entries from matrix X"""
        with torch.no_grad():
            ppreds = self.model.forward(torch.Tensor(X))

        nobs, nclasses = ppreds.shape

        if label_names is not None:
            assert (
                len(label_names) == nclasses
            ), "The number of label names and classes don't match."
            labeller = lambda i: label_names[i]
        else:
            labeller = lambda i: str(i)

        return [
            {labeller(i): float(ppreds[j][i]) for i in range(nclasses)}
            for j in range(nobs)
        ]
