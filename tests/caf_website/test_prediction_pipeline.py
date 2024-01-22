from caf_website.models.catalog import Classifier, Embedder
from caf_website.pipeline import predict


def test_prediction_pipeline_nnlm():
    prediction = predict("Esto es una frase de prueba", Embedder.NNLM, Classifier.LSVM)

    assert type(prediction) == dict
    assert type(list(prediction.keys())[0]) == str
    assert type(list(prediction.values())[0]) == float


def test_prediction_pipeline_w2v():
    prediction = predict("Esto es una frase de prueba", Embedder.W2V, Classifier.LSVM)

    assert type(prediction) == dict
    assert type(list(prediction.keys())[0]) == str
    assert type(list(prediction.values())[0]) == float


def test_prediction_pipeline_mlt():
    prediction = predict("Esto es una frase de prueba", Embedder.MLT, Classifier.LSVM)

    assert type(prediction) == dict
    assert type(list(prediction.keys())[0]) == str
    assert type(list(prediction.values())[0]) == float
