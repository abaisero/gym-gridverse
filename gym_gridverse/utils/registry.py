import abc
import inspect
from collections import UserDict
from typing import Callable, List, Optional


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
        """Register a function in this registry.

        This method can be either called directly or used as a decorator.
        Before registration, the function signature is checked to make sure it
        matches the appropriate protocol, and the name is checked to avoid
        conflicts.  If `name` is not given, `function.__name__` is used.

        Usage:

            >>> @registry.register
            >>> def function_1(...):
                    ...

            >>> @registry.register(name='alt_name_2')
            >>> def function_2(...):
                    ...

            >>> def function_3(...):
                    ...
            >>> registry.register(function_3)

            >>> def function_4(...):
                    ...
            >>> registry.register(function_4, name='alt_name_4')

        Args:
            function: (`Callable, optional`)
            name: (`str, optional`)
        """

        # check inputs
        if function is None and name is None:
            raise ValueError('register() needs `function` or `name` (or both)')

        # used as decorator
        if function is not None:
            if not callable(function):
                TypeError('registered value must be a Callable')

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
