
from caf_website.pipeline.cleaning import clean_sentence

def test_cleaning_pipeline():
    clean = clean_sentence("Esto es una frase de prueba")
    assert clean == "frase prueba"
