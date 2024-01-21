from caf_website.utils import parse_enum
from caf_website.models.catalog import Classifier
from pytest import raises

def test_parse_enum():
    parsed = parse_enum("MLP", Classifier)
    assert type(parsed) == Classifier

def test_parse_enum_invalid_value():
    with raises(ValueError):
        parse_enum("invalid", Classifier)
