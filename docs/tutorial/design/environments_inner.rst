Inner Environments
==================

Inner environments represent the logic of the environment dynamics purely in
OOP.  Its methods receive and return :py:class:`~gym_gridverse.action.Action`,
:py:class:`~gym_gridverse.state.State`,
:py:class:`~gym_gridverse.observation.Observation` objects directly.

.. autoclass:: gym_gridverse.envs.inner_env.InnerEnv
  :noindex:
  :no-members:

An inner environment has the following responsibilities:

* (re)set the initial state (as :py:class:`~gym_gridverse.state.State`),
* update the state (as :py:class:`~gym_gridverse.state.State`),
* return the observation (as :py:class:`~gym_gridverse.observation.Observation`),
* return the reward (as :py:class:`float`),
* return the terminal signal (as :py:class:`bool`).

An inner environment provides these functionalities using two types of
interfaces: a functional one and a non-functional one.

Functional Interface
--------------------

The functional interface ignores the environment's internal state, and require
the method callers to provide their own states:

.. automethod:: gym_gridverse.envs.inner_env.InnerEnv.functional_reset
  :noindex:

.. automethod:: gym_gridverse.envs.inner_env.InnerEnv.functional_step
  :noindex:

.. automethod:: gym_gridverse.envs.inner_env.InnerEnv.functional_observation
  :noindex:

Non-Functional Interface
------------------------

The non-functional interface uses the environment's internal state, and
internally refers back to the functional interface by providing that state:

.. automethod:: gym_gridverse.envs.inner_env.InnerEnv.reset
  :noindex:

.. automethod:: gym_gridverse.envs.inner_env.InnerEnv.step
  :noindex:

.. autoattribute:: gym_gridverse.envs.inner_env.InnerEnv.state
  :noindex:

.. autoattribute:: gym_gridverse.envs.inner_env.InnerEnv.observation
  :noindex:

The following figure depicts the inner working of a generic inner environment.

.. figure:: /figures/inner-env-design-dark.png
  :width: 60%
  :align: center
  :figclass: only-dark

  Schematic of the inner environment.

.. figure:: /figures/inner-env-design-light.png
  :width: 60%
  :align: center
  :figclass: only-light

  Schematic of the inner environment.


GridWorld
---------

:py:class:`~gym_gridverse.envs.inner_env.InnerEnv` is actually a pure
interface, in the sense that it provides no concrete implementation but only a
set of methods which other concrete classes should instantiate.  Currently, GV
only provides a single implementation of this interface
(:py:class:`~gym_gridverse.envs.gridworld.GridWorld`), which makes specific
assumptions about the implementation of the functional methods.  Technically,
other implementations can (and will, eventually) be provided, for which the
rest of this section would not necessarily hold.

.. autoclass:: gym_gridverse.envs.gridworld.GridWorld
  :noindex:
