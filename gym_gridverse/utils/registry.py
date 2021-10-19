import abc
import inspect
from collections import UserDict
from typing import Callable, List, Optional

from gym_gridverse.debugging import checkraise


class FunctionRegistry(UserDict, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        assert False

    def get_nonprotocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        protocol_parameters = self.get_protocol_parameters(signature)
        return [
            parameter
            for parameter in signature.parameters.values()
            if parameter not in protocol_parameters
        ]

    @abc.abstractmethod
    def check_signature(self, function: Callable):
        assert False

    def register(self, function=None, *, name: Optional[str] = None):

        # check inputs
        checkraise(
            lambda: function is not None or name is not None,
            ValueError,
            'FunctionRegistry.register() must receive either '
            '`function` or `name` (or both).',
        )

        # used as decorator
        if function is not None:
            checkraise(
                lambda: callable(function),
                TypeError,
                'registered value must be a Callable',
            )

            self.check_signature(function)

            if name is None:
                name = function.__name__

            if name in self.data:
                raise ValueError(f'registry already contains name `{name}`')

            self.data[name] = function
            return function

        # else, used to create a decorator
        def register_decorator(function):
            self.register(function, name=name)

        return register_decorator
