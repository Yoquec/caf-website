import pickle
from enum import Enum, auto
from typing import Any, Optional, Self
from warnings import warn

import pytorch_lightning as pl
import torch
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Series
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from ..types import Data, IBaseClassifier, INNClassifier, Label, Path


class LSTMParam(Enum):
    """Parameters for the LSTM model"""

    INPUT_SIZE = auto()
    OUTPUT_SIZE = auto()
    HIDDEN_DIM = auto()
    LEARNING_RATE = auto()
    NUM_LAYERS = auto()


class LSTM(INNClassifier, IBaseClassifier):
    def __init__(self, model) -> None:
        super().__init__()
        self.model: LitLSTM = model

    @staticmethod
    def new(params: dict[LSTMParam, Any]):
        """Constructor

        Args:
            params:
                Dictionary with model parameters.
                These MUST include the following keys:
                - input_size: Length of the features (1 for scalar features)
                - ouput_size: Output of the last layer of the network (number of labels)
                - hidden_dim: Size of the hidden state vector $h$.
                Optional keys:
                - num_layers: Number of stacked LSTM layers (default = 1)
                - learning_rate: Learning rate for the Adam optimizer (default = 0.001)

        Returns:
            The unfitted model with set parameters
        """
        return LSTM(
            LitLSTM(
                input_size=params[LSTMParam.INPUT_SIZE],
                output_size=params[LSTMParam.OUTPUT_SIZE],
                hidden_dim=params[LSTMParam.HIDDEN_DIM],
                num_layers=params.get(LSTMParam.NUM_LAYERS, 1),
                learning_rate=params.get(LSTMParam.LEARNING_RATE, 0.001),
            )
        )

    @staticmethod
    def load(filename: Path):
        """Load a pretrained model"""
        with open(filename, "rb") as file:
            return LSTM(pickle.load(file))

    def predict(
        self, X: Data, label_names: Optional[list[str]] = None
    ) -> list[dict[Label, float]]:
        """Predict entries from matrix X

        Args:
            X:
                Data entries.
                All types will be converted to torch.Tensor
                    DataFrames: First column will be used as the data
            label_names:
                List of label names

        Returns:
            Returns a Dicctionary with a set of labels and their
            assigned probabilities.
        """

        match X:
            case ndarray() as arr:
                data = torch.from_numpy(arr)
            case DataFrame() as df:
                data = torch.from_numpy(df.iloc[:, 0].to_numpy())
            case Series() as series:
                data = torch.from_numpy(series.to_numpy())
            case Tensor() as tensor:
                data = tensor
            case _:
                raise TypeError("Invalid type for X")

        with torch.no_grad():
            ppreds: Tensor = self.model.forward(data)

        nobs, nclasses = ppreds.shape

        if label_names is not None:
            assert (
                len(label_names) == nclasses
            ), "The number of label names and classes don't match."
            labeller = lambda i: label_names[i]
        else:
            labeller = lambda i: str(self.model.classes_[i])

        preds = [
            {labeller(i): float(ppreds[j][i]) for i in range(nclasses)}
            for j in range(nobs)
        ]
        return preds

    def fit(
        self: Self,
        trainloader: DataLoader,
        epochs: int,
        validloader: Optional[DataLoader] = None,
        **trainer_kwargs,
    ) -> Self:
        """Fit the model to data.

        Args:
            trainloader:
                DataLoader with the training data. Must have been created with
                a CafDataset.
            validloader:
                DataLoader with the validation data. Must have been created with
                a CafDataset. Remember to disable shuffling 👍.
            epochs:
                (Max) Number of epochs to train
            trainer_kwargs:
                Additional keyword arguments to pass to the PyTorch Lightning Trainer

        Returns:
            The trained model
        """
        trainer = pl.Trainer(min_epochs=2, max_epochs=epochs)
        trainer.fit(self.model, trainloader, validloader, **trainer_kwargs)
        return self

    def save(self: Self, filename: Path) -> None:
        """Save a loaded model using Pickle"""
        if not filename.endswith(".pkl"):
            warn("Consider naming your saved model file with the `.pkl` extension.")

        with open(filename, "wb") as file:
            pickle.dump(self.model, file)


class LitLSTM(pl.LightningModule):
    """LSTM RNN network using torch Lightning

    LSTM network that  outputs a multiclass probability distribution over the labels.
    It applies log-softmax to the output of the last layer and computes the loss using
    the Negative Log Likelihood Loss function.
    Modifies parameters using the Adam optimizer.

    Attributes:
        lstm: LSTM layer
        fc: Fully connected Linear layer
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        num_layers: int = 1,
        learning_rate: float = 0.001,
    ) -> None:
        """Constructor for the LSTM

        Args:
            input_size: Length of the features (1 for scalar features)
            output_size: Output of the last layer of the network (number of labels)
            hidden_dim: Size of the hidden state vector $h$.
            num_layers: Number of stacked LSTM layers (default = 1)
        """
        super().__init__()

        self._input_size = input_size
        self._hidden_dim = hidden_dim
        self._stacked_layers = num_layers
        self._learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_size)
        self._loss = nn.NLLLoss()

    def forward(self, x: Tensor, length_vector: Optional[NDArray] = None) -> Tensor:  # type: ignore
        """Forward pass through the network

        Args:
            x:
                Input tensor of shape (batch_size, seq_len, input_size). If in
                training, batch normalization needs to have been applied previously.
            length_vector:
                Vector containing the true length of each sequence in the batch.
                (Only used in training)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        train = length_vector is not None
        batch_size = x.shape[0]

        # Initialize empty hidden state vectors
        h0 = torch.zeros(self._stacked_layers, batch_size, self._hidden_dim)
        c0 = torch.zeros(self._stacked_layers, batch_size, self._hidden_dim)

        h_out, _ = self.lstm(x, (h0, c0))

        # Extract the last item from the hidden state vectors for use in the fully connected layer
        if train and self._input_size > 1:
            h_final = self._extract_last_hidden_state(h_out, length_vector)
        else:
            h_final = h_out[:, -1, :]

        assert h_final.shape == (batch_size, self._hidden_dim)

        fc_out = self.fc(h_final)

        return nn.functional.softmax(fc_out, dim=1)

    def _extract_last_hidden_state(self, x: Tensor, length_vector: NDArray) -> Tensor:
        """Extracts the last valid hidden state from batch nomalization

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            length_vector: Vector containing the true length of each sequence in the batch

        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Neat trick from https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
        return x[torch.arange(x.shape[0]), length_vector - 1, :]

    def __common_step(self, batch) -> Tensor:
        x, y, l = batch
        y_hat = self.forward(x, length_vector=l)
        return self._loss(y_hat, y)

    def training_step(self, batch, batch_idx):  # type: ignore
        loss = self.__common_step(batch)
        if batch_idx % 5 == 0:
            self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, _):  # type: ignore
        loss = self.__common_step(batch)
        self.log("val_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self._learning_rate)
