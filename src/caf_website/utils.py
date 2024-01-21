from functools import reduce
from enum import EnumType
from typing import Any

def compose(*fns):
    return reduce(lambda f, g: lambda x: g(f(x)), fns)

def parse_enum(value: Any, enum: EnumType) -> EnumType:
    for _, member in enum.__members__.items():
        if member.value == value:  # type: ignore
            return member
    else:
        raise ValueError(f"Invalid value {value} for enum {enum}")
