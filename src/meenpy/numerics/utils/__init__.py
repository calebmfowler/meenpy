from numpy import array as nparray, float64 as npfloat
from pandas import read_csv
from sympy import Matrix, symbols

def variables(names: str, variable_class: type | None = None, **kwargs):
    if variable_class != None:
        return symbols(names, cls=variable_class, **kwargs)
    else:
        return symbols(names, **kwargs)

def matrix(*args, **kwargs) -> Matrix:
    return Matrix(*args, **kwargs)