import re
from functools import reduce

import spacy
from spellchecker import SpellChecker

__all__ = ["clean_sentence"]

_nlp = spacy.load("es_core_news_sm")
_checker = SpellChecker(language="es")


def __compose(*fns):
    return reduce(lambda f, g: lambda x: f(g(x)), fns)


def _replace_chars(text: str) -> str:
    text = re.sub(r"[^\w\d ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _remove_all_digit_words_and_degrees(text: str) -> str:
    return " ".join(
        word for word in text.split() if not word.isdigit() and "º" not in word
    )


def _spacy_normalization(text: str) -> str:
    return " ".join(
        word.lemma_
        for word in _nlp(text)
        if not word.is_stop and not word.is_punct and word.text.strip() != ""
    )


def _spell_checking(text: str) -> str:
    return " ".join(
        _checker.correction(word) or word
        if not re.search(r"\d", word) and not all(char.isupper() for char in word)
        else word
        for word in text.split()
    )


__cleaning_pipeline = __compose(
    lambda x: x.lower(),
    _replace_chars,
    _remove_all_digit_words_and_degrees,
    _spacy_normalization,
    _spell_checking,
)


def clean_sentence(sentence: str) -> str | None:
    clean = __cleaning_pipeline(sentence)
    return clean.strip() or None
