=====================
Creating Environments
=====================

GV offers three ways to create environments:  Programmatically, using `OpenAI
Gym`_'s :py:func:`~gym.make`, and using `YAML`_ configuration files.  The
programmatic way is the most flexible, but also the most cumbersome, as it
requires design knowledge and ad-hoc code.  For now, we focus on the last two.

Using OpenAI Gym
================

Some predefined environments can be instantiated directly using
:py:func:`gym.make`, e.g.,::

  import gym
  import gym_gridverse

  env = gym.make('GV-FourRooms-7x7-v0')

The resulting environment object is an instance of
:py:class:`~gym_gridverse.gym.GymEnvironment`, which not only satisfied the gym
interface, but also provides additional utilities.

Using YAML
==========

While :py:func:`gym.make` is a very convenient way to instantiate
pre-registered environments, the main strength of GV is the ability to create
custom environments by combining existing and custom transition functions,
observation functions, reward functions, etc.  Because custom environments
cannot be pre-registered, it is not possible to instantiate them via
:py:func:`gym.make`.  Instead, we provide a separate way to define and
instantiate environments based on YAML configuration files.  The ``yaml/``
folder contains some example environments in the YAML format, e.g.,

* :download:`yaml/gv_empty.8x8.yaml </../yaml/gv_empty.8x8.yaml>`
* :download:`yaml/gv_four_rooms.9x9.yaml </../yaml/gv_four_rooms.9x9.yaml>`
* :download:`yaml/gv_nine_rooms.13x13.yaml </../yaml/gv_nine_rooms.13x13.yaml>`
* :download:`yaml/gv_crossing.7x7.yaml </../yaml/gv_crossing.7x7.yaml>`
* :download:`yaml/gv_keydoor.9x9.yaml </../yaml/gv_keydoor.9x9.yaml>`
* :download:`yaml/gv_dynamic_obstacles.7x7.yaml </../yaml/gv_dynamic_obstacles.7x7.yaml>`
* :download:`yaml/gv_teleport.7x7.yaml </../yaml/gv_teleport.7x7.yaml>`

To create an environment from YAML, you can use the factory function
:py:meth:`~gym_gridverse.envs.yaml.factory.factory_env_from_yaml`.  However,
:py:meth:`~gym_gridverse.envs.yaml.factory.factory_env_from_yaml` returns an
instance of :py:class:`~gym_gridverse.envs.inner_env.InnerEnv`, which does not
provide the same interface as :py:class:`gym.Env`.  The
:py:class:`~gym_gridverse.envs.inner_env.InnerEnv` environment (and other
relevant classes) will be explained later, in the design section;  for now, if
you want to load a YAML environment and interact with it using the gym
interface, you must wrap it with :py:class:`~gym_gridverse.gym.GymEnvironment`
as follows::

  from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
  from gym_gridverse.gym import outer_env_factory, GymEnvironment

  inner_env = factory_env_from_yaml('path/to/env.yaml')
  state_representation = make_state_representation(
      'default',
      inner_env.state_space,
  )
  observation_representation = make_observation_representation(
      'default',
      inner_env.observation_space,
  )
  outer_env = OuterEnv(
      inner_env,
      state_representation=state_representation,
      observation_representation=observation_representation,
  )
  env = GymEnvironment(outer_env)

.. tip::

  Script :download:`scripts/gv_viewer.py </../scripts/gv_viewer.py>` loads an
  environment expressed in the YAML format and provides manual controls for the
  agent; this is currently the recommended way to check whether a YAML file is
  properly formatted, and that the resulting environment behaves as expected::

    gv_viewer.py yaml/gv_nine_rooms.13x13.yaml

Schema
------

The schema for the YAML format is provided in the json-schema_ format (since
YAML is approximately a superset of JSON): :download:`schema.yaml
</../schema.yaml>`.

Broadly speaking, the fields of the YAML format describe the environment spaces
(state, action, and observation), as well as its functions (reset, reward,
transition, observation, and terminating).  For a full overview, we refer to
the provided schema and the example YAML files.

.. _OpenAI Gym: https://gym.openai.com/
.. _YAML: https://yaml.org
.. _json-schema: https://json-schema.org/
