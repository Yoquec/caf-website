import pickle
from enum import Enum, auto
from typing import Any, Optional, Self
from warnings import warn

import xgboost as xgb
from numpy import unique

from ..types import IClassifier, IBaseClassifier, Data, Label, Path


class XGBoostParam(Enum):
    """XGBoost Parameters

    All interesting parameters to be trained inside
    the XGBoost classifier.

    Please refer to
    https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
    for more information

    """

    # Trainable parameters
    eta = auto()
    max_depth = auto()
    gamma = auto()

    # Hardware parameters
    device = auto()
    nthread = auto()


class XGBoost(IClassifier, IBaseClassifier):
    """XGBoost Classifier

    Class that represents an XGBoost classifier as Angela recommended.

    Attributes:
        model: The XGBoost model

    Examples:
        ```python
        # Load a model from disk (using pickle)
        loaded_xgb = XGBoost.load("my_saved_model.pkl")

        # Create a new model
        new_xgb = XGBoost.new({
            XGBoostParam.max_depth = 2,
            XGBoostParam.eta = 1
        }).fit(X, y)

        new_xgboost_pred = new_xgb.predict(Xtest)
        ```
    """

    # WARNING: If the fit method is used a lot, consider adding a `fit_Dmatrix()`
    # method that doesn't convert dataframes into Dmatrix.

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        self.model: xgb.Booster | None = None
        self._params = {"objective": "multi:softprob", **params}

    @staticmethod
    def new(params: dict[XGBoostParam, Any]):
        """Constructor for a new XGBoost model

        Args:
            params: Dictionary with model parameters

        Returns:
            The initialized unfitted model with set parameters
        """
        return XGBoost({key.name: value for key, value in params.items()})

    @staticmethod
    def load(filename: Path):
        """Load a pretrained XGBoost object using pickle"""
        with open(filename, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def load_model(filename: Path):
        """Load a saved model

        Load a pretrained model using internal XGBoost methods
        NOTE: Using internal methods, self._param will be empty!

        Args:
            filename: Name of the file with the saved model

        Returns:
            An XGBoost object containing the loaded model

        Raises:
            RuntimeError: There is a model already loaded
        """
        model_obj = XGBoost({})
        model_obj.model = xgb.Booster({})
        model_obj.model.load_model(filename)
        return model_obj

    def save_model(self, filename: Path) -> None:
        """Save a loaded model

        Save a model in JSON format using internal XGBoost methods.
        WARNING: The filename MUST end in `.json`.

        Args:
            filename: Name of the file with the saved model

        Raises:
            RuntimeError: There is no model loaded
            RuntimeError: The filename specified does not end in .json
        """
        if self.model is None:
            raise RuntimeError("There is no trained model to save.")

        elif not filename.endswith(".json"):
            raise RuntimeError(f"The filename '{filename}' does not end in '.json'.")

        self.model.save_model(filename)

    def predict(
        self, X: Data, label_names: Optional[list[str]] = None
    ) -> list[dict[Label, float]]:
        """Predict entries from matrix X

        Returns:
            Returns a Dicctionary with a set of labels and their
            assigned probabilities.

        Raises:
            RuntimeError: There is no model loaded
        """

        if self.model is None:
            raise RuntimeError("There is no trained model to make predictions with.")

        dtest = xgb.DMatrix(X)
        ppreds = self.model.predict(dtest)

        nobs, nclasses = ppreds.shape

        if label_names is not None:
            assert (
                len(label_names) == nclasses
            ), "The number of label names and classes don't match."
            labeller = lambda i: label_names[i]
        else:
            labeller = lambda i: str(i)

        return [
            {labeller(i): ppreds[j][i] for i in range(nclasses)} for j in range(nobs)
        ]

    def fit(
        self,
        X: Data,
        y: Data,
        Xval: Optional[Data] = None,
        yval: Optional[Data] = None,
        num_round=10,
    ) -> Self:
        """Fit the model to data.

        Args:
            X: Entries.
            y: Labels.
            Xval: Optional validation entries. If set alongside `yval`, the model
                will report performance during training.
            yval: Optional validation labels.
            num_round: Number of training iterations.

        Returns:
            Fitted model

        Raises:
            RuntimeError: There is a model already loaded
        """
        if self.model is not None:
            raise RuntimeError("There is a model already loaded.")

        dtrain = xgb.DMatrix(X, label=y)
        self._params["num_class"] = len(unique(y))
        common_args = [self._params, dtrain, num_round]

        if Xval is not None and yval is not None:
            dtest = xgb.DMatrix(Xval, label=yval)
            self.model = xgb.train(
                *common_args,
                # Add validation info
                [(dtrain, "train"), (dtest, "eval")],
            )

        else:
            self.model = xgb.train(*common_args)

        return self

    def save(self, filename: Path) -> None:
        """Save a loaded model using Pickle"""
        if not filename.endswith(".pkl"):
            warn("Consider naming your saved model file with the `.pkl` extension.")

        with open(filename, "wb") as file:
            # Dump the entire object for XGBoost, it's API is wierd, as
            # the training function is not a method, but a function in the
            # library.
            pickle.dump(self, file)
