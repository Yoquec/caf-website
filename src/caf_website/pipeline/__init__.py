from typing import Any, List, Protocol, Tuple

import numpy as np
import torch.nn.functional as F
from flamenn.layers import PerceptronLayer
from ..utils import compose

from caf_website.models.classifiers.mlp import MLPParam

from ..models.catalog import Classifier, Embedder
from ..models.classifiers import linear_svm, xgboost, mlp
from ..models.embedders import multilingual_transformer, nnlm, word2vec
from ..models.types import Document, IBaseClassifier, Path
from .cleaning import clean_sentence

model_paths: dict[Tuple[Classifier, Embedder], Path] = {
    # NNLM embeddings
    (Classifier.LSVM, Embedder.NNLM): Path("models/linear_svm.pkl"),
    (Classifier.MLP, Embedder.NNLM): Path("models/mlp_state_dict_nnlm.pkl"),
    (Classifier.XGBOOST, Embedder.NNLM): Path("models/xgboost.pkl"),
}

mlp_configurations: dict[Embedder, dict[MLPParam, Any]] = {
    Embedder.NNLM: {
        MLPParam.INPUT_SIZE: 128,
        MLPParam.OUTPUT_SIZE: 12,
        MLPParam.LEARNING_RATE: 0.001,
        MLPParam.CRITERION: F.nll_loss,
        MLPParam.LAYERS: [
            PerceptronLayer(64, F.relu, False),
            PerceptronLayer(32, F.relu, False),
        ]
    },
}

class IEmbedder(Protocol):
    def embed(self, documents: List[Document]) -> list[np.ndarray[float, Any]]:
        ...

def predict(
    sentence: str, embedder: Embedder, classifier: Classifier
) -> dict[str, float]:
    """Run predictions on the sentence"""
    embedding: IEmbedder
    model: IBaseClassifier

    match embedder:
        case Embedder.MLT:
            embedding = multilingual_transformer.MultilingualTransformer()
        case Embedder.NNLM:
            embedding = nnlm.NNLM()
        case Embedder.W2V:
            embedding = word2vec.Word2Vec()
        case _:
            raise RuntimeError(f"Incorrect embedder: {embedder}")

    match classifier:
        case Classifier.LSVM:
            model = linear_svm.LinearSVM.load(model_paths[(classifier, embedder)])
        case Classifier.XGBOOST:
            model = xgboost.XGBoost.load(model_paths[(classifier, embedder)])
        case Classifier.MLP:
            model = mlp.MLP.load(
                model_paths[(classifier, embedder)],
                mlp_configurations[embedder]
            )
        case _:
            raise RuntimeError(f"Incorrect classifier: {classifier}")

    pipeline = compose(
        clean_sentence,
        embedding.embed,
        model.predict,
    ) 

    return pipeline(sentence)
