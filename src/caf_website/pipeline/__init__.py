from typing import Any, List, Protocol, Tuple

import numpy as np
import torch.nn.functional as F
from flamenn.layers import PerceptronLayer

from caf_website.models.classifiers.mlp import MLPParam

from ..models.catalog import Classifier, Embedder
from ..models.classifiers import linear_svm, mlp, xgboost
from ..models.embedders import multilingual_transformer, nnlm, word2vec
from ..models.types import Document, IBaseClassifier, Path
from ..utils import compose
from .cleaning import clean_sentence

model_paths: dict[Tuple[Classifier, Embedder], Path] = {
    # NNLM embeddings
    (Classifier.LSVM, Embedder.NNLM): Path("models/svm_nnlm.pkl"),
    (Classifier.MLP, Embedder.NNLM): Path("models/mlp_state_dict_nnlm.pkl"),
    (Classifier.XGBOOST, Embedder.NNLM): Path("models/xgboost_internal_nnlm.json"),
    # MLT embeddings
    (Classifier.LSVM, Embedder.MLT): Path("models/svm_mlt.pkl"),
    (Classifier.MLP, Embedder.MLT): Path("models/mlp_state_dict_mlt.pkl"),
    (Classifier.XGBOOST, Embedder.MLT): Path("models/xgboost_internal_mlt.json"),
    # W2V embeddings
    (Classifier.LSVM, Embedder.W2V): Path("models/svm_w2v.pkl"),
    (Classifier.MLP, Embedder.W2V): Path("models/mlp_state_dict_w2v.pkl"),
    (Classifier.XGBOOST, Embedder.W2V): Path("models/xgboost_internal_w2v.json"),
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
        ],
    },
    Embedder.MLT: {
        MLPParam.INPUT_SIZE: 512,
        MLPParam.OUTPUT_SIZE: 12,
        MLPParam.LEARNING_RATE: 0.001,
        MLPParam.CRITERION: F.nll_loss,
        MLPParam.LAYERS: [
            PerceptronLayer(256, F.relu, False),
            PerceptronLayer(128, F.relu, False),
            PerceptronLayer(64, F.relu, False),
            PerceptronLayer(32, F.relu, False),
        ],
    },
    Embedder.W2V: {
        MLPParam.INPUT_SIZE: 300,
        MLPParam.OUTPUT_SIZE: 12,
        MLPParam.LEARNING_RATE: 0.001,
        MLPParam.CRITERION: F.nll_loss,
        MLPParam.LAYERS: [
            PerceptronLayer(128, F.relu, False),
            PerceptronLayer(64, F.relu, False),
            PerceptronLayer(32, F.relu, False),
        ],
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
            model = xgboost.XGBoost.load_model(model_paths[(classifier, embedder)])
        case Classifier.MLP:
            model = mlp.MLP.load(
                model_paths[(classifier, embedder)], mlp_configurations[embedder]
            )
        case _:
            raise RuntimeError(f"Incorrect classifier: {classifier}")

    pipeline = compose(
        lambda t: clean_sentence(t),
        lambda t: embedding.embed([t]),
        lambda t: model.predict(t),
        lambda t: t[0],
    )

    return pipeline(sentence)
