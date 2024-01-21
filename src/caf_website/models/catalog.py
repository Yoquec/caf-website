from enum import Enum

from ..utils import parse_enum


class Classifier(Enum):
    MLP = "MLP"
    XGBOOST = "XGBoost"
    LSVM = "Linear SVM"


class Embedder(Enum):
    MLT = "Multilingual Transformer"
    NNLM = "NNLM"
    W2V = "Word2Vec"


def parse_classifier(value: str) -> Classifier:
    return parse_enum(value, Classifier)  # type: ignore


def parse_embedder(value: str) -> Embedder:
    return parse_enum(value, Embedder)  # type: ignore
