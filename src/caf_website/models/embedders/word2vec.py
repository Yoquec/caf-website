import logging
import os
from collections.abc import Iterable
from pickle import PickleError
from typing import Literal, overload

import numpy as np
import py7zr
from gensim.models import KeyedVectors
from gensim.models import Word2Vec as Word2VecModel

from ..types import Document

DEFAULT_MODEL_PATH = "./static/SBW-vectors-300-min5.kv"


class Word2Vec:
    @overload
    def __init__(
        self,
        path_or_dataset: str = DEFAULT_MODEL_PATH,
        train: Literal[False] = False,
        word_embeddings: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        path_or_dataset: Iterable[str],
        train: Literal[True],
        word_embeddings: bool = False,
    ) -> None:
        ...

    def __init__(
        self,
        path_or_dataset: str | Iterable[str] = DEFAULT_MODEL_PATH,
        train: bool = False,
        word_embeddings: bool = False,
    ) -> None:
        self._logger = logging.getLogger(type(self).__name__)
        self._word_embeddings = word_embeddings

        if not train:
            try:
                self._model: KeyedVectors = KeyedVectors.load(path_or_dataset)
            except PickleError:
                self._model: KeyedVectors = KeyedVectors.load_word2vec_format(
                    path_or_dataset
                )
            except FileNotFoundError as fe:
                # If no file found, try to extract it from the 7z file
                if isinstance(path_or_dataset, str):
                    directory = os.path.dirname(path_or_dataset)
                    archive = f"{path_or_dataset}.7z"

                    if os.path.basename(archive) in os.listdir(directory):
                        with py7zr.SevenZipFile(archive, "r") as a:
                            self._logger.info(
                                f"Extracting '{archive}' to '{directory}'."
                            )
                            a.extractall(directory)

                        self._model: KeyedVectors = KeyedVectors.load(path_or_dataset)
                    else:
                        raise FileNotFoundError(
                            f"Archive '{archive}' not found in folder '{directory}'."
                        ) from fe
                else:
                    raise fe
        else:
            self._model = Word2VecModel(sentences=path_or_dataset).wv

    def embed(
        self, documents: list[Document]
    ) -> list[np.ndarray[float, np.dtype[np.float32]]]:
        vec_shape = self._model["a"].shape
        doc_vectors = []

        for doc in documents:
            word_vectors = []

            for word in doc.split():
                if word in self._model:
                    word_vectors.append(self._model[word])
                else:
                    self._logger.warning(
                        "Word '%s' not present in the Word2Vec model", word
                    )
                    if self._word_embeddings:
                        word_vectors.append(np.zeros(vec_shape, dtype=np.float32))

            if not word_vectors:
                self._logger.warning("No words in document were recognized: %s", doc)
                doc_vectors.append(np.zeros(vec_shape, dtype=np.float32))
            elif self._word_embeddings:
                doc_vectors.append(np.stack(word_vectors, dtype=np.float32))
            else:
                doc_vectors.append(
                    np.mean(np.stack(word_vectors), axis=0, dtype=np.float32)
                )

        return doc_vectors
