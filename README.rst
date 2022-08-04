=============
gym-gridverse
=============


.. image:: https://img.shields.io/pypi/v/gym-gridverse.svg
        :target: https://pypi.python.org/pypi/gym-gridverse

.. image:: https://github.com/abaisero/gym-gridverse/actions/workflows/build.yml/badge.svg
        :target: https://github.com/abaisero/gym-gridverse/actions/workflows/build.yml

.. image:: https://readthedocs.org/projects/gym-gridverse/badge/?version=latest
        :target: https://gym-gridverse.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Gridworld domains for fully and partially observable reinforcement learning


* Free software: MIT license
* Documentation: https://gym-gridverse.readthedocs.io.


Features
--------

Customization
"""""""""""""

GridVerse_ is highly customizable;  while many components are provided
out-of-the-box, it is designed such that you can create your own components
programmatically, including your own objects, starting states, transition
functions, reward functions, observation functions, terminating functions, etc.

The following `GridObjects` are provided:

* :code:`Floor` -- An empty tile.
* :code:`Wall` -- An opaque wall.
* :code:`Exit` -- An exit tile.
* :code:`Door` -- A door which can be opened/closed.
* :code:`Key` -- An item to open a locked `Door`.
* :code:`MovingObstacle` -- An obstacle which moves autonomously.
* :code:`Box` -- A container of other `GridObjects`.
* :code:`Telepod` -- A teleporting tile.

The following transition functions are provided:

* :code:`move_agent` -- Moves the agent.
* :code:`turn_agent` -- Turns the agent.
* :code:`pickndrop` -- Lets agent pick and/or drop an object.
* :code:`actuate_door` -- Opens/closes a Door.
* :code:`actuate_box` -- Opens a Box.
* :code:`move_obstacles` -- Lets MovingObstacle objects move.
* :code:`teleport` -- Teleports the agent across the Telepods.

The following reward functions are provided:

* :code:`reduce_sum` -- A sum of other rewards
* :code:`living_reward` -- A constant reward
* :code:`reach_exit` -- A reward for reaching an Exit.
* :code:`overlap` -- A reward for standing on/off a GridObject type.
* :code:`proportional_to_distance` -- Reward based on distance from a GridObject type.
* :code:`getting_closer` -- Rewards for moving closer to/further from a GridObject type.
* :code:`actuate_door` -- Rewards for actuating a Door.
* :code:`pickndrop` -- Rewards for picking and/or dropping GridObject types.

The following observation functions are provided:

* :code:`from_visibility` -- Observability determined by custom visibility functions.
* :code:`full_observation` -- Observability which is unblocked by Walls.
* :code:`partial_observation` -- Observability which is blocked by Walls.
* :code:`raytracing observation` -- Observability determined by direct line of sight.

The following terminating functions are provided:

* :code:`reduce_any` -- Terminates if any of the given terminating functions are satisfied.
* :code:`reduce_all` -- Terminates if all of the given terminating functions are satisfied.
* :code:`overlap` -- Terminates if the agent is standing on a GridObject type.
* :code:`reach_exit` -- Terminates if the agent reaches an Exit.
* :code:`bump_moving_obstacle` -- Terminates if the agent bumps into a MovingObstacle.
* :code:`bump_into_wall` -- Terminates if the agent bumps into a Wall.

YAML Configuration Files
""""""""""""""""""""""""

Aside being able to define your own environments programmatically, GridVerse_
allows you to create and share YAML configuration files which fully describe
the components which define an environment.  This is a very convenient way to
create an environment made of existing components and share it with the world.
The `yaml/` folder contains a number of environments defined using the YAML
configuration format.

Suitable for Fully/Partially Observable Control Problems for Learning/Planning
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Depending on your research interests, most GridVerse_ components can be used to
form either fully observable or partially observable control problems.
Further, GridVerse_ environments provide both a state-ful and a functional
interface, depending on whether you are addressing learning or planning
problems.

Future work / in progress:
""""""""""""""""""""""""""

* 100\% test coverage
* Multi-agent support
* Benchmark performance of reinforcement learning and planning algorithms

Examples
""""""""

+---------------------------------------------------------------------------------------------------+
| yaml/gv_crossing.7x7.yaml                                                                         |
+================================================+==================================================+
| State                                          | Observations                                     |
|                                                |                                                  |
| |gv_crossing.7x7.state.gif|                    | |gv_crossing.7x7.observation.montage.gif|        |
+------------------------------------------------+--------------------------------------------------+

.. |gv_crossing.7x7.state.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_crossing.7x7.state.gif?raw=true
.. |gv_crossing.7x7.observation.montage.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_crossing.7x7.observation.montage.gif?raw=true

+--------------------------------------------------------------------------------------------------------+
| yaml/gv_dynamic_obstacles.7x7.yaml                                                                     |
+================================================+=======================================================+
| State                                          | Observations                                          |
|                                                |                                                       |
| |gv_dynamic_obstacles.7x7.state.gif|           | |gv_dynamic_obstacles.7x7.observation.montage.gif|    |
+------------------------------------------------+-------------------------------------------------------+

.. |gv_dynamic_obstacles.7x7.state.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_dynamic_obstacles.7x7.state.gif?raw=true
.. |gv_dynamic_obstacles.7x7.observation.montage.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_dynamic_obstacles.7x7.observation.montage.gif?raw=true

+---------------------------------------------------------------------------------------------------+
| yaml/gv_empty.8x8.yaml                                                                            |
+================================================+==================================================+
| State                                          | Observations                                     |
|                                                |                                                  |
| |gv_empty.8x8.state.gif|                       | |gv_empty.8x8.observation.montage.gif|           |
+------------------------------------------------+--------------------------------------------------+

.. |gv_empty.8x8.state.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_empty.8x8.state.gif?raw=true
.. |gv_empty.8x8.observation.montage.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_empty.8x8.observation.montage.gif?raw=true

+---------------------------------------------------------------------------------------------------+
| yaml/gv_four_rooms.9x9.yaml                                                                       |
+================================================+==================================================+
| State                                          | Observations                                     |
|                                                |                                                  |
| |gv_four_rooms.9x9.state.gif|                  | |gv_four_rooms.9x9.observation.montage.gif|      |
+------------------------------------------------+--------------------------------------------------+

.. |gv_four_rooms.9x9.state.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_four_rooms.9x9.state.gif?raw=true
.. |gv_four_rooms.9x9.observation.montage.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_four_rooms.9x9.observation.montage.gif?raw=true

+---------------------------------------------------------------------------------------------------+
| yaml/gv_keydoor.5x5.yaml                                                                          |
+================================================+==================================================+
| State                                          | Observations                                     |
|                                                |                                                  |
| |gv_keydoor.5x5.state.gif|                     | |gv_keydoor.5x5.observation.montage.gif|         |
+------------------------------------------------+--------------------------------------------------+

.. |gv_keydoor.5x5.state.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_keydoor.5x5.state.gif?raw=true
.. |gv_keydoor.5x5.observation.montage.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_keydoor.5x5.observation.montage.gif?raw=true

+---------------------------------------------------------------------------------------------------+
| yaml/gv_nine_rooms.13.13.yaml                                                                     |
+================================================+==================================================+
| State                                          | Observations                                     |
|                                                |                                                  |
| |gv_nine_rooms.13x13.state.gif|                | |gv_nine_rooms.13x13.observation.montage.gif|    |
+------------------------------------------------+--------------------------------------------------+

.. |gv_nine_rooms.13x13.state.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_nine_rooms.13x13.state.gif?raw=true
.. |gv_nine_rooms.13x13.observation.montage.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_nine_rooms.13x13.observation.montage.gif?raw=true

+---------------------------------------------------------------------------------------------------+
| yaml/gv_teleport.7x7.yaml                                                                         |
+================================================+==================================================+
| State                                          | Observations                                     |
|                                                |                                                  |
| |gv_teleport.7x7.state.gif|                    | |gv_teleport.7x7.observation.montage.gif|        |
+------------------------------------------------+--------------------------------------------------+

.. |gv_teleport.7x7.state.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_teleport.7x7.state.gif?raw=true
.. |gv_teleport.7x7.observation.montage.gif| image:: https://github.com/abaisero/gym-gridverse/blob/master/images/yaml/gv_teleport.7x7.observation.montage.gif?raw=true


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
others,  e.g., MiniGrid_ includes tasks which involve natural language
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

.. [1] While Minigrid_ provides :code:`FullyObsWrapper`, which extends the
  agent's observation range, it does not represents true full-state
  observability.

.. _GridVerse: https://github.com/abaisero/gym-gridverse
.. _MiniGrid: https://github.com/maximecb/gym-minigrid
.. _MiniWorld: https://github.com/maximecb/gym-miniworld

Citation
--------

If you use `gym-gridverse`, please cite it:

.. code-block:: bibtex

  @misc{baisero2021gym-gridverse,
      author = {Andrea Baisero and Sammie Katt},
      title = {gym-gridverse: Gridworld domains for fully and partially observable reinforcement learning},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/abaisero/gym-gridverse}},
  }

Credits
-------

This package was inspired by MiniGrid_, and created with Cookiecutter_ and the
`audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
