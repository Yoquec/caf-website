from enum import Enum

class Classifier(Enum):
    MLP = "MLP"
    XGBOOST = "XGBoost"
    LSVM = "Linear SVM"

class Embedder(Enum):
    MLT = "Multilingual Transformer"
    NNLM = "NNLM"
    W2V = "Word2Vec"
