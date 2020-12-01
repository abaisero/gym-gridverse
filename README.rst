=============
gym-gridverse
=============


.. image:: https://img.shields.io/pypi/v/gym-gridverse.svg
        :target: https://pypi.python.org/pypi/gym-gridverse

.. image:: https://img.shields.io/travis/abaisero/gym_gridverse.svg
        :target: https://travis-ci.com/abaisero/gym_gridverse

.. image:: https://readthedocs.org/projects/gym-gridverse/badge/?version=latest
        :target: https://gym-gridverse.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Gridworld domains for fully and partially observable reinforcement learning


* Free software: MIT license
* Documentation: https://gym-gridverse.readthedocs.io.


Features
--------

* TODO


Similar Projects
----------------

The GridVerse_ project takes heavy inspiration from MiniGrid_, and was designed
to address a few shortcomings which limited our ability to it fully:

Customization and Configurability
  Our design philosophy is primarily based on user customization.  We provide
  interfaces for you to design your own objects, state dynamics, reward
  functions, observability, etc.  We also provide a YAML-based configuration
  format which will allow you to conveniently share environmens with others.

Time-Invariant Reward Functions
  Our reward functions satisfy the formal time-invariance property of Markov
  decision processes.

Full Observability
  We provide a full observability interface which satisfies the formal
  property of Markov decision processes.

Functional Interface
  We provide a functional interface which enables the use of planning methods,
  e.g., MCTS, POMCP.

MiniWorld_ is a 3D variant similar to MiniGrid_ by the same authors.

While GridVerse_ provides functionality which we found useful and/or necessary
for our needs, each project provides something which is unique compared to the
others,  e.g., Minigrid_ includes tasks which involve natural language
comprehension, and MiniWorld_ incorporates a whole third dimension.  Make sure
to browse all projects to get a clearer picture on which best suits your needs.

.. |check| unicode:: U+2714 .. check mark
.. |cross| unicode:: U+2718 .. cross mark

.. csv-table:: Project Comparison
  :header:  ,                       GridVerse_, MiniGrid_,  MiniWorld_

            2D Environments,        |check|,    |check|,    ""
            3D Environments,        "",         "",         |check|
            Partial Observability,  |check|,    |check|,    |check|
            Full Observability,     |check|,    [1]_,        ""
            RGB Observability,      "",         |check|,    |check|
            Natural Language Tasks, "",         |check|,    ""
            Customizable,           |check|,    "",         |check|
            YAML-Configurable,      |check|,    "",         ""

.. [1] While :py:class:`gym_minigrid.wrappers.FullyObsWrapper` extends the
   observation range, it does not represents true full-state observability.

.. _GridVerse: https://github.com/abaisero/gym-gridverse
.. _MiniGrid: https://github.com/maximecb/gym-minigrid
.. _MiniWorld: https://github.com/maximecb/gym-miniworld


Credits
-------

This package was inspired by MiniGrid_, and created with Cookiecutter_ and the
`audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
