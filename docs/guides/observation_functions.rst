=====================
Observation Functions
=====================

In this section we describe the observation function protocol, the observation
functions provided in :py:mod:`~gym_gridverse.envs.observation_functions`, and
how to write your own custom observation functions.

The ObservationFunction Protocol
================================

An observation function is a generative function which represents a (stochastic)
mapping from a state to an observation.  Using the :py:mod:`typing`
standard library, the observation function type is defined as a
:py:class:`typing.Protocol` with a :py:meth:`__call__` member which receives a
:py:class:`~gym_gridverse.state.State` and an optional
:py:class:`numpy.random.Generator`, and returns an
:py:class:`~gym_gridverse.observation.Observation`.

.. autoclass:: gym_gridverse.envs.observation_functions.ObservationFunction
  :noindex:
  :members: __call__

.. note::
  An observation function may (and often does) accept additional arguments;
  this is possible **so long as** the extra arguments either have default
  values, or are binded to specific values later on, e.g., using
  :py:func:`functools.partial`.

Provided Observation Functions
==============================

The :py:mod:`~gym_gridverse.envs.observation_functions` module contains some
predefined observation functions, among which:

- :py:func:`~gym_gridverse.envs.observation_functions.from_visibility` -- uses
  a :py:data:`~gym_gridverse.envs.visibility_functions.VisibilityFunction` to
  determine which tiles are visible.

- :py:func:`~gym_gridvserse.envs.observation_functions.full_observation` --
  every tile is visible;  implemented via
  :py:func:`~gym_gridverse.envs.observation_functions.full_visibility`.

- :py:func:`~gym_gridvserse.envs.observation_functions.minigrid_observation` --
  the observation used by the :py:mod:`gym_minigrid` package;  implemented via
  :py:func:`~gym_gridverse.envs.visibility_functions.minigrid_visibility`.

- :py:func:`~gym_gridverse.envs.observation_functions.raytracing_observation`
  -- observation is determined by direct unobstructed line of sight from the
  agent's tile;  implemented via
  :py:func:`~gym_gridverse.envs.visibility_functions.raytracing_visibility`.

Custom Observation Functions
============================

.. note::
  While writing your own observation function directly is indeed a
  possibility, the preferred way to implement new types of observations is by
  writing a custom
  :py:data:`~gym_gridverse.envs.visibility_functions.VisibilityFunction` and
  using it with the
  :py:func:`~gym_gridverse.envs.observation_functions.from_visibility`
  observation function.

Custom observation functions can be defined so long as they satisfy some basic
rules;  A custom observation function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.observation_functions.ObservationFunction`
  protocol.

- **SHOULD** use the :py:data:`rng` argument as the source for any
  stochasticity.

- **MUST** use :py:func:`~gym_gridverse.rng.get_gv_rng_if_none` (only if the
  :py:data:`rng` is used at all).

.. warning::
  The :py:data:`rng` argument is used to control the source of randomness and
  allow for the environment to be seeded via
  :py:meth:`~gym_gridverse.envs.env.Environment.set_seed`, which in turn
  guarantees the reproducibility of traces, runs, and experiments;  if you wish
  to use external sources of randomness, you will have to manage them and their
  seeding yourself.

Practical Example
-----------------

In this example, we are going to write an observation function which will
restrict the agent's view to only the tiles in front, and only until the first
non-transparent tile.

.. literalinclude:: example__observation_function__frontal_line_of_sight.py
  :language: python
