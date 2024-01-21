from typing import Any, List, Self

import numpy as np
import tensorflow_hub as hub
from numpy.typing import NDArray

from ..types import Document


class NNLM:
    """
    Singleton class that represents the NNLM transformer
    model for Spanish taken from tensorflow hub.

    Model link: https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1

    - The model preprocesses its input by removing punctuation and splitting on spaces.
    - Word embeddings are combined into sentence embedding using the sqrtn combiner
    - The least frequent tokens and embeddings (~2.5%) are replaced by hash buckets

    Example usage:
    ```
    nnlm = NNLM()
    embeddings = nnlm.embed([
        "Esto es un perro.", # Document 1
        "Hay gatos y perros en esta casa." # Document 2
        ])
    ```
    """

    __instance = None
    _model = None

    def __new__(cls) -> Self:
        if cls.__instance is None:
            cls.__instance = super(NNLM, cls).__new__(cls)
            cls.__instance._model = cls._download_model()
        return cls.__instance

    @staticmethod
    def _download_model():
        """
        Method that downloads the NNLM model from tensorflow_hub if needed
        and loads it into memory.

        NOTE: This method is not intended to be used outside the class
        constructor
        """
        print("Downloading the NNLM model and loading it into memory ⏳")
        return hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")

    def embed(self, documents: List[Document]) -> list[np.ndarray[float, Any]]:
        """
        Transfoms each document into a vector representation
        sentence embedding of size 128.

        Example code:
        ```
        nnlm = NNLM()
        embeddings = nnlm.embed([
            "Esto es un perro.", # Document 1
            "Hay gatos y perros en esta casa." # Document 2
            ])

        embeddings.shape
        # >>> (2, 128)
        ```
        """
        assert self._model is not None, "Model not loaded"

        vectors: NDArray = self._model(documents).numpy()
        return [vector for vector in vectors]  # type: ignore
