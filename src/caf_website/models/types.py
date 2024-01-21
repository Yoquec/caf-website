from typing import Any, Optional, Protocol, Self

from numpy import float64, int32
from numpy.typing import NDArray
from pandas import DataFrame, Series
from torch.utils.data import DataLoader

Document = str
Label = str
Path = str
Data = Series | DataFrame | NDArray[float64]


class IBaseClassifier(Protocol):
    @staticmethod
    def new(params) -> Any:
        """Create a new model"""
        ...

    @staticmethod
    def load(filename: Path) -> Any:
        """Load a pretrained model"""
        ...

    def predict(self: Self, X: Data) -> list[dict[Label, float]]:
        """Predict entries from matrix X

        Returns:
            Returns a Dicctionary with a set of labels and their
            assigned probabilities.
        """
        ...

    def save(self: Self, filename: Path) -> None:
        """Save a model"""
        ...


class IClassifier(Protocol):
    """Protocol that defines the fit method for Non Neural Network models"""

    def fit(self: Self, X: Data, y: Data) -> Self:
        """Fit the model to data.

        Args:
            X: Entries
            y: Labels

        Returns:
            The trained model
        """
        ...


class INNClassifier(Protocol):
    """Protocol that defines the fit method for Neural Network models"""

    def fit(
        self: Self,
        trainloader: DataLoader,
        epochs: int,
        validloader: Optional[DataLoader] = None,
    ) -> Self:
        """Fit the model to data.

        Args:
            trainloader: DataLoader with the training data
            validloader: DataLoader with the validation data
            epochs: Number of epochs to train

        Returns:
            The trained model
        """
        ...



