from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

import imageio.v2 as iio
import more_itertools as mitt
import numpy as np
from typing_extensions import TypedDict

from gym_gridverse.action import Action
from gym_gridverse.observation import Observation
from gym_gridverse.rendering import GridVerseViewer
from gym_gridverse.state import State
from gym_gridverse.utils.rl import make_return_computer

Image: TypeAlias = np.ndarray
"""An image, alias to np.ndarray"""

FrameType = TypeVar('FrameType', State, Observation, np.ndarray)
"""A State, Observation, or image (np.ndarray)"""


@dataclass(frozen=True)
class Data(Generic[FrameType]):
    """Data for recordings of states or observations"""

    frames: Sequence[FrameType]
    actions: Sequence[Action]
    rewards: Sequence[float]
    discount: float

    def __post_init__(self):
        if not len(self.frames) - 1 == len(self.actions) == len(self.rewards):
            raise ValueError('wrong lengths')

    @property
    def is_state_data(self) -> bool:
        return isinstance(self.frames[0], State)

    @property
    def is_observation_data(self) -> bool:
        return isinstance(self.frames[0], Observation)

    @property
    def is_image_data(self) -> bool:
        return isinstance(self.frames[0], Image)


@dataclass(frozen=True)
class DataBuilder(Generic[FrameType]):
    """Builds Data object interactively"""

    frames: List[FrameType] = field(init=False, default_factory=list)
    actions: List[Action] = field(init=False, default_factory=list)
    rewards: List[float] = field(init=False, default_factory=list)
    discount: float

    def append0(self, frame: FrameType):
        if len(self.frames) != 0:
            raise RuntimeError('cannot call DataBuilder.append0 at this point')

        self.frames.append(frame)

    def append(self, frame: FrameType, action: Action, reward: float):
        if len(self.frames) == 0:
            raise RuntimeError('cannot call DataBuilder.append at this point')

        self.frames.append(frame)
        self.actions.append(action)
        self.rewards.append(reward)

    def build(self) -> Data[FrameType]:
        return Data(self.frames, self.actions, self.rewards, self.discount)


class HUD_Info(TypedDict):
    action: Optional[Action]
    reward: Optional[float]
    ret: Optional[float]
    done: Optional[bool]


def generate_images(
    data: Union[Data[State], Data[Observation], Data[Image]]
) -> Iterator[Image]:
    """Generate images associated with the input data"""

    if data.is_image_data:
        yield from data.frames
        return

    data = cast(Union[Data[State], Data[Observation]], data)
    shape = data.frames[0].grid.shape
    viewer = GridVerseViewer(shape)
    viewer.flip_hud()

    hud_info: HUD_Info = {
        'action': None,
        'reward': None,
        'ret': None,
        'done': None,
    }

    yield viewer.render(data.frames[0], return_rgb_array=True, **hud_info)

    return_computer = make_return_computer(data.discount)

    for _, is_last, (frame, action, reward) in mitt.mark_ends(
        zip(data.frames[1:], data.actions, data.rewards)
    ):
        hud_info = {
            'action': action,
            'reward': reward,
            'ret': return_computer(reward),
            'done': is_last,
        }

        frame = cast(Union[State, Observation], frame)
        yield viewer.render(frame, return_rgb_array=True, **hud_info)

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
        if filenames is None:
            raise ValueError(f'invalid arguments for mode {mode}')

        record_images(filenames, images, **kwargs)

    if mode == 'gif':
        if filename is None:
            raise ValueError(f'invalid arguments for mode {mode}')

        record_gif(filename, images, **kwargs)

    if mode == 'mp4':
        if filename is None:
            raise ValueError(f'invalid arguments for mode {mode}')

        record_mp4(filename, images, **kwargs)


def record_images(
    filenames: Iterable[str],
    images: Sequence[np.ndarray],
    **kwargs,
):
    """Create image files from input images"""

    for filename, image in zip(filenames, images):
        print(f'creating {filename}')
        try:
            iio.imwrite(filename, image)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            iio.imwrite(filename, image)


def record_gif(
    filename: str,
    images: Sequence[np.ndarray],
    *,
    loop: int = 0,
    fps: float = 2.0,
    duration: Optional[float] = None,
    **kwargs,
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
        iio.mimwrite(filename, images, **kwargs)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        iio.mimwrite(filename, images, **kwargs)


def record_mp4(
    filename: str,
    images: Sequence[np.ndarray],
    *,
    fps: float = 2.0,
    duration: Optional[float] = None,
    **kwargs,
):
    """Create an mp4 file from input images"""

    kwargs = {
        'format': 'mp4',
        'fps': fps,
    }

    if duration is not None:
        kwargs['fps'] = len(images) / duration

    print(f'creating {filename} ({len(images)} frames)')
    try:
        iio.mimwrite(filename, images, **kwargs)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        iio.mimwrite(filename, images, **kwargs)
