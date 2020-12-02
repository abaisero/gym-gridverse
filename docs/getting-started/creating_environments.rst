=====================
Creating Environments
=====================

`Gridverse` is a library for constructing and developing environments. In this
section we cover how to use YAML_ to define and create them.

Play with GUI
=============

The interactive GUI gives an idea of the features provided in this package. The
script expects a single input, the `YAML` file describing the environment, for
example::

  >> scripts/gv_viewer.py envs_yaml/env_minigrid_four_rooms.yaml

Other environments provided out of the box are located in the directory
`envs_yaml`, and include:

- :download:`envs_yaml/env_minigrid_crossing.yaml <../../envs_yaml/env_minigrid_crossing.yaml>`
- :download:`envs_yaml/env_minigrid_door_key.yaml <../../envs_yaml/env_minigrid_door_key.yaml>`
- :download:`envs_yaml/env_minigrid_dynamic_obstacles_16x16_random.v0.yaml <../../envs_yaml/env_minigrid_dynamic_obstacles_16x16_random.v0.yaml>`
- :download:`envs_yaml/env_minigrid_empty.yaml <../../envs_yaml/env_minigrid_empty.yaml>`
- :download:`envs_yaml/env_minigrid_four_rooms.yaml <../../envs_yaml/env_minigrid_four_rooms.yaml>`

Create environments
===================

To use environments in your own code, you can call the factory function with a
path to the `YAML` configurations::

  from gym_gridverse.envs.factory_yaml import make_environment
  env = make_environment(path_to_yaml_file)

.. todo::

  A seamless integration with the GYM_ interface is provided....

YAML scheme
===========

.. todo::

  Describe fields somehow

.. _YAML: https://yaml.org/
.. _GYM: https://gym.openai.com/
