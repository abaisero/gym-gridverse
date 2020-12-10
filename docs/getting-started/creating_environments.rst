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

  >> scripts/gv_viewer.py envs_yaml/gv_four_rooms.13x13.yaml

Other environments provided out of the box are located in the directory
`envs_yaml`, and include:

- :download:`yaml/gv_empty.8x8.yaml <../../yaml/gv_empty.8x8.yaml>`
- :download:`yaml/gv_four_rooms.9x9.yaml <../../yaml/gv_four_rooms.9x9.yaml>`
- :download:`yaml/gv_nine_rooms.13x13.yaml <../../yaml/gv_nine_rooms.13x13.yaml>`
- :download:`yaml/gv_crossing.7x7.yaml <../../yaml/gv_crossing.7x7.yaml>`
- :download:`yaml/gv_keydoor.9x9.yaml <../../yaml/gv_keydoor.9x9.yaml>`
- :download:`yaml/gv_dynamic_obstacles.7x7.yaml <../../yaml/gv_dynamic_obstacles.7x7.yaml>`
- :download:`yaml/gv_teleport.7x7.yaml <../../yaml/gv_teleport.7x7.yaml>`

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
