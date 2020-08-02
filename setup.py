from setuptools import setup

from gym_gridverse import __version__

setup(
    name='gym-gridverse',
    version=__version__,
    description='Gridworld domains in the gym interface',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/gym-gridverse',
    packages=['gym_gridverse'],
    install_requires=['gym', 'more_itertools', 'numpy', 'termcolor'],
    license='MIT',
)
