from caf_website.models.catalog import Classifier, Embedder
from caf_website.pipeline import predict


def test_prediction_pipeline_types():
    prediction = predict("Esto es una frase de prueba", Embedder.NNLM, Classifier.LSVM)

    assert type(prediction) == dict
    assert type(list(prediction.keys())[0]) == str
    assert type(list(prediction.values())[0]) == float
