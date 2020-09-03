from typing import Dict

import numpy as np
from gym_gridverse.representations.representation import Representation


class DefaultObservationRepresentation(Representation):
    """The default representation for observations

    Simply returns the observation as indices

    """

    def __init__(self):
        pass

    @property
    def space(self) -> Dict[str, np.ndarray]:
        pass

    def convert(self, x) -> Dict[str, np.ndarray]:
        pass


class CompactObservationRepresentation(Representation):
    """Returns observations as indices but 'not sparse'

    Will jump over unused indices to allow for smaller spaces

    """

    def __init__(self):
        pass

    @property
    def space(self) -> Dict[str, np.ndarray]:
        pass

    def convert(self, x) -> Dict[str, np.ndarray]:
        pass


def create_observation_representation() -> Representation:
    """Factory function for observation representations

    TODO: nyi, current returns `ObservationToArray`

    Returns:
        Representation: [TODO:description]
    """
    return DefaultObservationRepresentation()
