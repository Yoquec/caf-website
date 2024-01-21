from typing import Any, List, Self

import numpy as np
import tensorflow_hub as hub
import tensorflow_text as _
from numpy.typing import NDArray

from ..types import Document


class MultilingualTransformer:
    """
    Singleton class that represents the multilingual
    sentence encoder model taken from tensorflow hub.

    Model link:
        https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3

    - Covers 16 languages, showing strong performance on cross-lingual retrieval.
    - The input to the module is variable length text in any of the
    aforementioned languages and the output is a 512 dimensional vector.
    - Input text can have arbitrary length! However, model
    time and space complexity is $$O(n^2)$$ for input length $$n$$.
    We recommend inputs that are approximately one sentence in length.

    Example usage:
    ```
    mult_transformer = MultilingualTranformer()
    embeddings = mult_transformer.embed([
        "Esto es un perro.", # Document 1
        "Hay gatos y perros en esta casa." # Document 2
        ])
    ```
    """

    __instance = None
    _model = None

    def __new__(cls) -> Self:
        if cls.__instance is None:
            cls.__instance = super(MultilingualTransformer, cls).__new__(cls)
            cls.__instance._model = cls._download_model()
        return cls.__instance

    @staticmethod
    def _download_model():
        """
        Method that downloads the MultilingualTranformer model from tensorflow_hub if needed
        and loads it into memory.

        NOTE: This method is not intended to be used outside the class
        constructor
        """
        print(
            "Downloading the multilingual sentence encoder and loading it into memory ⏳"
        )
        return hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
        )

    def embed(self, documents: List[Document]) -> list[np.ndarray[float, Any]]:
        """
        Transfoms each document into a vector representation
        sentence embedding of size 128.

        Example code:
        ```
        mult_transformer = MultilingualTranformer()
        embeddings = mult_transformer.embed([
            "Esto es un perro.", # Document 1
            "Hay gatos y perros en esta casa." # Document 2
            ])

        embeddings.shape
        # >>> TensorShape([2, 128])
        ```
        """
        assert self._model is not None

        vectors: NDArray = self._model(documents).numpy()
        return [vector for vector in vectors]  # type: ignore
