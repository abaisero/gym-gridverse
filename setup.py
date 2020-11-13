from setuptools import setup

from gym_gridverse import __version__

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='gym-gridverse',
    version=__version__,
    description='Gridworld domains for reinforcement learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/gym-gridverse',
    packages=['gym_gridverse'],
    install_requires=[
        'gym',
        'more_itertools',
        'numpy',
        'pyglet',
        'termcolor',
        'yamale',
    ],
    package_data={'gym_gridverse': ['envs/resources/schema.yaml']},
    scripts=['scripts/gv_gym_interface.py', 'scripts/gv_yaml.py'],
    license='MIT',
)
