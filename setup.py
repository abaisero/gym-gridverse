from setuptools import setup

from gym_gridverse import __version__

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='gym-gridverse',
    version=__version__,
    description='Gridworld domains for fully and partially observable reinforcement learning',
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
        'pytest',
        'termcolor',
        'yamale',
    ],
    package_data={'gym_gridverse': ['envs/resources/schema.yaml']},
    scripts=['scripts/gv_gym_interface.py', 'scripts/gv_yaml.py'],
    license='MIT',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Science/Research",
    ],
    python_requires='~=3.6',
)
