====================
Transition Functions
====================

In this section we describe the transition function protocol, the transition
functions provided in :py:mod:`~gym_gridverse.envs.transition_functions`, and
how to write your own custom transition functions.

The TransitionFunction Protocol
===============================

A transition function is a generative function which represents a (stochastic)
mapping from a state-action pair to a next states.  Using the :py:mod:`typing`
standard library, the transition function type is defined as a
:py:class:`typing.Protocol` with a :py:meth:`__call__` member which receives a
:py:class:`~gym_gridverse.state.State`, an
:py:class:`gym_gridverse.action.Action`, and an optional
:py:class:`numpy.random.Generator`, and edits the input
:py:class:`~gym_gridverse.state.State`.

.. autoclass:: gym_gridverse.envs.transition_functions.TransitionFunction
    :noindex:
    :members: __call__

.. note::
    A transition function may (and often does) accept additional arguments;
    this is possible **so long as** the extra arguments either have default
    values, or are binded to specific values later on, e.g., using
    :py:func:`functools.partial`.

Provided Transition Functions
=============================

The :py:mod:`~gym_gridverse.envs.transition_functions` module contains some
predefined transition functions, among which:

- :py:func:`~gym_gridverse.envs.transition_functions.update_agent` -- moves and
  turns the agent.

- :py:func:`~gym_gridverse.envs.transition_functions.step_moving_obstacles` -- executes
  object dynamics.

- :py:func:`~gym_gridverse.envs.transition_functions.step_telepod` -- executes
  object dynamics.

- :py:func:`~gym_gridverse.envs.transition_functions.actuate_door` --
  executes object actuation.

- :py:func:`~gym_gridverse.envs.transition_functions.pickup_mechanics` -- picks
  up and drops down objects.

Custom Transition Functions
===========================

Custom transition functions can be defined so long as they satisfy some basic
rules;  A custom transition function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.transition_functions.TransitionFunction`
  protocol.

- **MUST** edit the input state, rather than return a new state altogether.

- **SHOULD** use the :py:data:`rng` argument as the source for any
  stochasticity.

- **MUST** use :py:func:`~gym_gridverse.rng.get_gv_rng_if_none` (only if the
  :py:data:`rng` is used at all).

.. warning::
  The :py:data:`rng` argument is used to control the source of randomness and
  allow for the environment to be seeded via
  :py:meth:`~gym_gridverse.envs.inner_env.InnerEnv.set_seed`, which in turn
  guarantees the reproducibility of traces, runs, and experiments;  if you wish
  to use external sources of randomness, you will have to manage them and their
  seeding yourself.

Practical Example 1
-------------------

In this example, we are going to write a transition function in which at every
time-step one of the Floor tiles turns into a Wall tile.

.. literalinclude:: example__transition_function__creeping_walls.py
  :language: python

Practical Example 2
-------------------

In this example, we are going to write a transition function in which the agent
moves until it hits an obstacle.

.. literalinclude:: example__transition_function__rooklike_movement.py
  :language: python

Practical Example 3
-------------------

In this example, we are going to write a transition function which randomizes
the execution of another transition function.

.. literalinclude:: example__transition_function__random_transition.py
  :language: python
