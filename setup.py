#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'cached_property',
    'gym',
    'more_itertools',
    'numpy',
    'pyyaml',
    'schema',
    'termcolor',
    'typing-extensions',  # python3.7 compatibility
]

setup(
    author="Andrea Baisero",
    author_email='andrea.baisero@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    description="Gridworld domains for fully and partially observable reinforcement learning",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='gym_gridverse',
    name='gym_gridverse',
    packages=find_packages(include=['gym_gridverse', 'gym_gridverse.*']),
    scripts=[
        'scripts/gv_gym_interface.py',
        'scripts/gv_yaml_schema.py',
        'scripts/gv_yaml.py',
        'scripts/gv_viewer.py',
    ],
    url='https://github.com/abaisero/gym-gridverse',
    version='0.0.1',
    zip_safe=False,
)
