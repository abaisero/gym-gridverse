================
Reset Functions
================

In this section we describe the reset function protocol, the reset functions
provided in :py:mod:`~gym_gridverse.envs.reset_functions`, and how to write
your own custom reset functions.

The ResetFunction Protocol
===========================

A reset function is a generative function which represents a (stochastic)
distribution over initial states.  Using the :py:mod:`typing` standard library,
the reset function type is defined as a :py:class:`typing.Protocol` with a
:py:meth:`__call__` member which receives an optional
:py:class:`numpy.random.Generator` and returns a
:py:class:`~gym_gridverse.state.State`.

.. autoclass:: gym_gridverse.envs.reset_functions.ResetFunction
    :noindex:
    :members: __call__

.. note::
    A reset function may (and often does) accept additional arguments;  this
    is possible **so long as** the extra arguments either have default values,
    or are binded to specific values later on, e.g., using
    :py:func:`functools.partial`.

Provided Reset Functions
=========================

The :py:mod:`~gym_gridverse.envs.reset_functions` module contains some
predefined reset functions, among which:

- :py:func:`~gym_gridverse.envs.reset_functions.reset_minigrid_empty` -- a
  room with a :py:class:`~gym_gridverse.grid_object.Goal`.

- :py:func:`~gym_gridverse.envs.reset_functions.reset_minigrid_rooms` --
  connected rooms with a :py:class:`~gym_gridverse.grid_object.Goal`.

- :py:func:`~gym_gridverse.envs.reset_functions.reset_minigrid_keydoor` -- two
  rooms connected by a locked :py:class:`~gym_gridverse.grid_object.Door`; on
  one side is a :py:class:`~gym_gridverse.grid_object.Key`, and on the other
  side a :py:class:`~gym_gridverse.grid_object.Goal`.
    
- :py:func:`~gym_gridverse.envs.reset_functions.reset_minigrid_dynamic_obstacles`
  -- a room with a :py:class:`~gym_gridverse.grid_object.Goal` and many
  :py:class:`~gym_gridverse.grid_object.MovingObstacle`.

Custom Reset Functions
======================

Custom reset functions can be defined so long as they satisfy some basic rules;  A
custom reset function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.reset_functions.ResetFunction` protocol.

- **SHOULD** use the :py:data:`rng` argument as the source for any
  stochasticity.

- **MUST** use :py:func:`~gym_gridverse.rng.get_gv_rng_if_none` (only if the
  :py:data:`rng` is used at all).

.. warning:: The :py:data:`rng` argument is used to control the source of
   randomness and allow for the environment to be seeded via
   :py:meth:`~gym_gridverse.envs.inner_env.InnerEnv.set_seed`, which in turn
   guarantees the reproducibility of traces, runs, and experiments;  if you
   wish to use external sources of randomness, you will have to manage them and
   their seeding yourself.

Practical Example 1
-------------------

In this example, we are going to write an extremely simple reset function in
which the agent is positioned in front of a
:py:class:`~gym_gridverse.grid_object.Goal`.

.. literalinclude:: example__reset_function__simplest.py
  :language: python

Practical Example 2
-------------------

In this example, we are going to write a reset function for an environment in
which the agent needs to pick up the correctly colored
:py:class:`~gym_gridverse.grid_object.Key` to open a randomly colored
:py:class:`~gym_gridverse.grid_object.Door` and reach the
:py:class:`~gym_gridverse.grid_object.Goal`.

.. literalinclude:: example__reset_function__match_key_color.py
  :language: python
