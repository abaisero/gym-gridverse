=====================
Creating Environments
=====================

Gridverse is a library for constructing and instantiating custom gridworld
environments. In this section, we cover how to load predefined environments
using `OpenAI Gym`_, and how to create custom environments using `YAML`_.

Using OpenAI Gym
================

Some predefined environments, e.g. ``MiniGrid-FourRooms-v0``, can be
instantiated directly using :py:func:`gym.make`::

  import gym
  import gym_gridverse

  env = gym.make('MiniGrid-FourRooms-v0')

However, the strength of Gridverse is the ability to create custom environments
by combining transition functions, observation functions, reward functions,
etc.  Because custom environments cannot be pre-registered, it is not possible
to instantiate them via :py:func:`gym.make`.  Instead, we provide a separate
way to define and instantiate environments based on YAML configuration files.

Using YAML
==========

The YAML configuration format allows you to specify all aspects of an
environment.  The ``yaml/`` folder contains some predefined environments in the
YAML format, e.g.,

- :download:`yaml/gv_empty.8x8.yaml <../../yaml/gv_empty.8x8.yaml>`
- :download:`yaml/gv_four_rooms.9x9.yaml <../../yaml/gv_four_rooms.9x9.yaml>`
- :download:`yaml/gv_nine_rooms.13x13.yaml <../../yaml/gv_nine_rooms.13x13.yaml>`
- :download:`yaml/gv_crossing.7x7.yaml <../../yaml/gv_crossing.7x7.yaml>`
- :download:`yaml/gv_keydoor.9x9.yaml <../../yaml/gv_keydoor.9x9.yaml>`
- :download:`yaml/gv_dynamic_obstacles.7x7.yaml <../../yaml/gv_dynamic_obstacles.7x7.yaml>`
- :download:`yaml/gv_teleport.7x7.yaml <../../yaml/gv_teleport.7x7.yaml>`

GUI with Manual Control
-----------------------

Script :download:`scripts/gv_viewer.py <../../scripts/gv_viewer.py>` loads an
environment expressed in the YAML format and provides manual controls, and is
also currently the favorite way to check whether a YAML file is properly
formatted::

  >> gv_viewer.py yaml/gv_nine_rooms.13x13.yaml

Instantiating from YAML
-----------------------

To create an environment from YAML, you can use the factory function
:py:meth:`~gym_gridverse.envs.yaml.factory.factory_env_from_yaml`::

  from gym_gridverse.envs.yaml.factory import factory_env_from_yaml

  env = factory_env_from_yaml(path_to_yaml_file)  # type: InnerEnv

.. note::

  :py:meth:`~gym_gridverse.envs.yaml.factory.factory_env_from_yaml` returns an
  instance of :py:class:`~gym_gridverse.envs.inner_env.InnerEnv`, which does
  not provide the same interface as :py:class:`gym.core.Env`.  If you want to
  load a custom environment specified from YAML and interact with it using the
  gym interface, you can use :py:class:`~gym_gridverse.gym.GymEnvironment`::

    from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
    from gym_gridverse.gym import GymEnvironment

    # returns gym_gridverse.envs.inner_env.InnerEnv
    inner_env = factory_env_from_yaml(path_to_yaml_file)  

    # returns gym_gridverse.gym.GymEnvironment (which inherits gym.Env)
    gym_env = GymEnvironment.from_environment(inner_env)

YAML schema
-----------

The schema for the YAML format is provided in the json-schema_ format (since
YAML is approximately a superset of JSON): :download:`schema.yaml
<../../schema.yaml>`.

Broadly speaking, the fields of the YAML format describe the environment spaces
(state, action, and observation), as well as its functions (reset, reward,
transition, observation, and terminating).  For a full overview, we refer to
the provided schema and the example YAML files.

.. _OpenAI Gym: https://gym.openai.com/
.. _YAML: https://yaml.org
.. _json-schema: https://json-schema.org/
