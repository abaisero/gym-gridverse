import os
from dataclasses import dataclass, field
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import imageio
import more_itertools as mitt
import numpy as np
from typing_extensions import TypedDict

from gym_gridverse.action import Action
from gym_gridverse.observation import Observation
from gym_gridverse.rendering import GridVerseViewer
from gym_gridverse.state import State
from gym_gridverse.utils.rl import make_return_computer

Element = TypeVar('Element', State, Observation, np.ndarray)


@dataclass(frozen=True)
class Data(Generic[Element]):
    """Data for recordings of states or observations"""

    elements: Sequence[Element]
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

    @property
    def is_image_data(self):
        return isinstance(self.elements[0], np.ndarray)


@dataclass(frozen=True)
class DataBuilder(Generic[Element]):
    """Builds Data object interactively"""

    elements: List[Element] = field(init=False, default_factory=list)
    actions: List[Action] = field(init=False, default_factory=list)
    rewards: List[float] = field(init=False, default_factory=list)
    discount: float

    def append0(self, element: Element):
        if len(self.elements) != 0:
            raise RuntimeError('cannot call DataBuilder.append0 at this point')

        self.elements.append(element)

    def append(self, element: Element, action: Action, reward: float):
        if len(self.elements) == 0:
            raise RuntimeError('cannot call DataBuilder.append at this point')

        self.elements.append(element)
        self.actions.append(action)
        self.rewards.append(reward)

    def build(self) -> Data[Element]:
        return Data(self.elements, self.actions, self.rewards, self.discount)


class HUD_Info(TypedDict):
    action: Optional[Action]
    reward: Optional[float]
    ret: Optional[float]
    done: Optional[bool]


def generate_images(data: Data[Element]) -> Iterator[np.ndarray]:
    """Generate images associated with the input data"""

    if data.is_image_data:
        yield from data.elements
        return

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
        try:
            imageio.imwrite(filename, image)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            imageio.imwrite(filename, image)


def record_gif(
    filename: str,
    images: Sequence[np.ndarray],
    *,
    loop: int = 0,
    fps: float = 2.0,
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
    try:
        imageio.mimwrite(filename, images, **kwargs)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        imageio.mimwrite(filename, images, **kwargs)
