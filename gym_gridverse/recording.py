import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Union

import imageio
import more_itertools as mitt
import numpy as np
from typing_extensions import TypedDict

from gym_gridverse.action import Action
from gym_gridverse.observation import Observation
from gym_gridverse.rendering import GridVerseViewer
from gym_gridverse.state import State
from gym_gridverse.utils.rl import make_return_computer


@dataclass(frozen=True)
class Data:
    """Data for recordings of states or observations"""

    elements: Union[Sequence[State], Sequence[Observation]]
    actions: Sequence[Action]
    rewards: Sequence[float]
    discount: float

    def __post_init__(self):
        if not len(self.elements) - 1 == len(self.actions) == len(self.rewards):
            raise ValueError('wrong lengths')

    @property
    def is_state_data(self):
        return isinstance(self.elements[0], State)

    @property
    def is_observation_data(self):
        return isinstance(self.elements[0], Observation)


class HUD_Info(TypedDict):
    action: Optional[Action]
    reward: Optional[float]
    ret: Optional[float]
    done: Optional[bool]


def generate_images(data: Data) -> Iterator[np.ndarray]:
    """Generate images associated with the input data"""

    shape = data.elements[0].grid.shape
    viewer = GridVerseViewer(shape)

    hud_info: HUD_Info = {
        'action': None,
        'reward': None,
        'ret': None,
        'done': None,
    }

    yield viewer.render(data.elements[0], return_rgb_array=True, **hud_info)

    return_computer = make_return_computer(data.discount)

    for _, is_last, (element, action, reward) in mitt.mark_ends(
        zip(data.elements[1:], data.actions, data.rewards)
    ):
        hud_info = {
            'action': action,
            'reward': reward,
            'ret': return_computer(reward),
            'done': is_last,
        }

        yield viewer.render(element, return_rgb_array=True, **hud_info)  # type: ignore

    viewer.close()


def record(
    mode: str,
    images: Sequence[np.ndarray],
    *,
    filename: Optional[str] = None,
    filenames: Optional[Iterable[str]] = None,
    **kwargs,
):
    """Factory function for other recording functions"""

    if mode == 'images':
        if None in [filenames]:
            raise ValueError(f'invalid arguments for mode {mode}')

        assert filenames is not None  # forces typing
        record_images(filenames, images, **kwargs)

    if mode == 'gif':
        if None in [filename]:
            raise ValueError(f'invalid arguments for mode {mode}')

        assert filename is not None  # forces typing
        record_gif(filename, images, **kwargs)


def record_images(
    filenames: Iterable[str],
    images: Sequence[np.ndarray],
    **kwargs,  # pylint: disable=unused-argument
):
    """Create image files from input images"""

    for filename, image in zip(filenames, images):
        print(f'creating {filename}')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        imageio.imwrite(filename, image)


def record_gif(
    filename: str,
    images: Sequence[np.ndarray],
    *,
    loop: int = 0,
    fps: float,
    duration: Optional[float] = None,
    **kwargs,  # pylint: disable=unused-argument
):
    """Create a gif file from input images"""

    kwargs = {
        'format': 'gif',
        'subrectangles': True,
        'loop': loop,
        'fps': fps,
    }

    if duration is not None:
        kwargs['duration'] = duration / len(images)

    print(f'creating {filename} ({len(images)} frames)')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimwrite(filename, images, **kwargs)
