=====================
Terminating Functions
=====================

In this section we describe the terminating function protocol, the terminating
function registry, and how to write your own custom terminating functions.

The TerminatingFunction Protocol
================================

A terminating function is a deterministic mapping from a state-action-state
transition to a boolean indicating the termination of an episode.  Using the
:py:mod:`typing` standard library, the terminating function type is defined as
a :py:class:`typing.Callable` which receives a
:py:class:`~gym_gridverse.state.State`, an
:py:class:`~gym_gridverse.action.Action`, and another
:py:class:`~gym_gridverse.state.State`, and returns a :py:class:`bool`.

.. autodata:: gym_gridverse.envs.terminating_functions.TerminatingFunction
   :noindex:

.. note::
  A terminating function may (and often does) accept additional arguments;
  this is possible **so long as** the extra arguments either have default
  values, or are binded to specific values later on, e.g., using
  :py:func:`functools.partial`.

The Terminating Function Registry
=================================

The :py:mod:`~gym_gridverse.envs.terminating_functions` module contains some
pre-defined terminating functions and the
:py:data:`~gym_gridverse.envs.terminating_functions.terminating_function_registry`,
a dictionary-like object through which to register and retrieve terminating
functions.  Terminating functions are registered using the
:py:meth:`~gym_gridverse.utils.registry.FunctionRegistry.register` method,
which can be used as a decorator (also see
:ref:`tutorial/customization/terminating_functions:Custom Terminating
Functions`).  As a dictionary,
:py:data:`~gym_gridverse.envs.terminating_functions.terminating_function_registry`
has a :py:meth:`~dict.keys` method which returns the names of registered
functions.

Custom Terminating Functions
============================

Custom terminating functions can be defined so long as they satisfy some basic rules;  A
custom terminating function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.terminating_functions.TerminatingFunction`
  protocol.

- **MUST NOT** edit the input states.

- **SHOULD** be wholly deterministic.

Practical Example 1
-------------------

.. note::
  The examples shown here can be found in the ``examples/`` folder.

In this example, we are going to write a terminating function in which the
episode terminates if the agent has not changed from the previous time-step,
i.e., it's position, orientation, and object held are the same.

.. literalinclude:: /../examples/static_agent_terminating.py
  :language: python

Practical Example 2
-------------------

In this example, we are going to write a terminating function in which the
episode terminates if the agent does not observe a given object;  this could be
an interesting way to encode a "escort mission" task, where the agent needs to
keep an eye on a moving target.

.. note::
    The following example has a couple of minor issues.  First of all, the
    observation used in the terminating function might not be the same received
    by the agent;  this could happen if the observation_function itself is
    different, or if it's stochastic.  Second, because the
    :py:data:`~gym_gridverse.envs.terminating_functions.TerminatingFunction`
    protocol does not have an ``rng`` argument, this terminating function does
    not allow to fully reproduce executions, if the given observation function
    is stochastic.

.. literalinclude:: /../examples/concealed_terminating.py
  :language: python
