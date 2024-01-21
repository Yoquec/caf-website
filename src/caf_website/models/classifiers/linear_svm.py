import pickle
from enum import Enum, auto
from typing import Any, Optional, Self
from warnings import warn

from sklearn import svm

from ..types import IClassifier, IBaseClassifier, Data, Label, Path


class SVMParam(Enum):
    C = auto()


class LinearSVM(IClassifier, IBaseClassifier):
    """SVM with linear kernel

    Class that represents the SVM with linear kernel that Angela recommended.

    Attributes:
        model: SVM model

    Examples:
        ```python
        # Load a model from disk
        loaded_svm = LinearSVM.load("my_saved_model.pkl")

        # Create a new model
        new_svm = LinearSVM.new({
            SVMParam.C = 0.8
            }).fit(X, y)
        new_svm_pred = new_svm.predict(Xtest)
        ```
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model: svm.SVC = model

    @staticmethod
    def new(params: dict[SVMParam, Any]):
        """Constructor

        Args:
            params: Dictionary with model parameters

        Returns:
            The unfitted model with set parameters
        """
        return LinearSVM(
            svm.SVC(
                kernel="linear",
                probability=True,
                **{key.name: value for key, value in params.items()}
            )
        )

    @staticmethod
    def load(filename: Path):
        """Load a pretrained model"""
        with open(filename, "rb") as file:
            return LinearSVM(pickle.load(file))

    def predict(
        self, X: Data, label_names: Optional[list[str]] = None
    ) -> list[dict[Label, float]]:
        """Predict entries from matrix X

        Returns:
            Returns a Dicctionary with a set of labels and their
            assigned probabilities.
        """

        ppreds = self.model.predict_proba(X)
        nobs, nclasses = ppreds.shape

        if label_names is not None:
            assert (
                len(label_names) == nclasses
            ), "The number of label names and classes don't match."
            labeller = lambda i: label_names[i]
        else:
            labeller = lambda i: str(self.model.classes_[i])

        preds = [
            {labeller(i): ppreds[j][i] for i in range(nclasses)} for j in range(nobs)
        ]
        return preds

    def fit(self, X: Data, y: Data) -> Self:
        """Fit the model to data.

        Args:
            X: Entries
            y: Labels

        Returns:
            Fitted model
        """
        self.model.fit(X, y)
        return self

    def save(self, filename: Path) -> None:
        """Save a loaded model using Pickle"""
        if not filename.endswith(".pkl"):
            warn("Consider naming your saved model file with the `.pkl` extension.")

        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
