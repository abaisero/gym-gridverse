import inspect
import itertools as itt
from typing import List


def get_positional_parameters(
    signature: inspect.Signature, n: int
) -> List[inspect.Parameter]:
    try:
        return list(itt.islice(signature.parameters.values(), n))
    except ValueError as error:
        raise TypeError(f'signature needs {n} positional argument') from error


def get_keyword_parameter(
    signature: inspect.Signature, name: str
) -> inspect.Parameter:
    try:
        return signature.parameters[name]
    except KeyError as error:
        raise TypeError('signature needs `{name}` keyword argument') from error
