#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pyglet<=1.5.27',
    'gym<=0.21.0',
    'imageio',
    'imageio-ffmpeg',
    'more_itertools',
    'numpy>=1.20.0',
    'pyyaml',
    'schema',
    'typing-extensions',
]

setup(
    author="Andrea Baisero",
    author_email='andrea.baisero@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    description="Customizable gridworld domains for fully and partially observable reinforcement learning and planning",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='gym_gridverse',
    name='gym_gridverse',
    packages=find_packages(include=['gym_gridverse', 'gym_gridverse.*']),
    package_data={'gym_gridverse': ['py.typed', 'registered_envs/*.yaml']},
    scripts=[
        'scripts/gv_control_loop_gym.py',
        'scripts/gv_control_loop_inner.py',
        'scripts/gv_control_loop_outer.py',
        'scripts/gv_profile.py',
        'scripts/gv_record.py',
        'scripts/gv_viewer.py',
        'scripts/gv_yaml.py',
        'scripts/gv_yaml_schema.py',
    ],
    url='https://github.com/abaisero/gym-gridverse',
    version='0.0.1',
    zip_safe=False,
)
