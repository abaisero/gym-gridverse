====================
Transition Functions
====================

In this section we describe the transition function protocol, the transition
function registry, and how to write your own custom transition functions.

The TransitionFunction Protocol
===============================

A transition function is a generative function which represents a (stochastic)
mapping from a state-action pair to a next states.  Using the :py:mod:`typing`
standard library, the transition function type is defined as a callable
:py:class:`typing.Protocol` which receives a
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

The Transition Function Registry
================================

The :py:mod:`~gym_gridverse.envs.transition_functions` module contains some
pre-defined transition functions and the
:py:data:`~gym_gridverse.envs.transition_functions.transition_function_registry`,
a dictionary-like object through which to register and retrieve transition
functions.  Transition functions are registered using the
:py:meth:`~gym_gridverse.utils.registry.FunctionRegistry.register` method,
which can be used as a decorator (also see
:ref:`tutorial/customization/transition_functions:Custom Transition
Functions`).  As a dictionary,
:py:data:`~gym_gridverse.envs.transition_functions.transition_function_registry`
has a :py:meth:`~dict.keys` method which returns the names of registered
functions.

Custom Transition Functions
===========================

Custom transition functions can be defined so long as they satisfy some basic
rules;  A custom transition function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.transition_functions.TransitionFunction`
  protocol.

- **MUST** edit the input state, rather than return a new state altogether.

- **SHOULD** use the ``rng`` argument as the source for any stochasticity.

- **MUST** use :py:func:`~gym_gridverse.rng.get_gv_rng_if_none` (only if the
  ``rng`` is used at all).

.. warning::
  The ``rng`` argument is used to control the source of randomness and allow
  for the environment to be seeded via
  :py:meth:`~gym_gridverse.envs.inner_env.InnerEnv.set_seed`, which in turn
  guarantees the reproducibility of traces, runs, and experiments;  if you wish
  to use external sources of randomness, you will have to manage them and their
  seeding yourself.

Practical Example 1
-------------------

.. note::
  The examples shown here can be found in the ``examples/`` folder.

In this example, we are going to write a transition function in which at every
time-step one of the Floor tiles turns into a Wall tile.

.. literalinclude:: /../examples/creeping_walls_transition.py
  :language: python

Practical Example 2
-------------------

In this example, we are going to write a transition function in which the agent
moves until it hits an obstacle.

.. literalinclude:: /../examples/chessrook_movement_transition.py
  :language: python

Practical Example 3
-------------------

In this example, we are going to write a transition function which randomizes
the execution of another transition function.

.. literalinclude:: /../examples/random_transition.py
  :language: python
