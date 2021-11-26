===============
Reset Functions
===============

In this section we describe the reset function protocol, the reset function
registry, and how to write your own custom reset functions.

The ResetFunction Protocol
==========================

A reset function is a generative function which represents a (stochastic)
distribution over initial states.  Using the :py:mod:`typing` standard library,
the reset function type is defined as a callable :py:class:`typing.Protocol`
which receives an optional :py:class:`numpy.random.Generator` and returns a
:py:class:`~gym_gridverse.state.State`.

.. autoclass:: gym_gridverse.envs.reset_functions.ResetFunction
    :noindex:
    :members: __call__

.. note::
    A reset function may (and often does) accept additional arguments;  this
    is possible **so long as** the extra arguments either have default values,
    or are binded to specific values later on, e.g., using
    :py:func:`functools.partial`.

The Reset Function Registry
===========================

The :py:mod:`~gym_gridverse.envs.reset_functions` module contains some
pre-defined reset functions and the
:py:data:`~gym_gridverse.envs.reset_functions.reset_function_registry`, a
dictionary-like object through which to register and retrieve reset functions.
Reset functions are registered using the
:py:meth:`~gym_gridverse.utils.registry.FunctionRegistry.register` method,
which can be used as a decorator (also see
:ref:`tutorial/customization/reset_functions:Custom Reset Functions`).  As a
dictionary,
:py:data:`~gym_gridverse.envs.reset_functions.reset_function_registry` has a
:py:meth:`~dict.keys` method which returns the names of registered functions.

Custom Reset Functions
======================

Custom reset functions can be defined so long as they satisfy some basic rules;  A
custom reset function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.reset_functions.ResetFunction` protocol.

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

In this example, we are going to write an extremely simple reset function in
which the agent is positioned in front of a
:py:class:`~gym_gridverse.grid_object.Exit`.

.. literalinclude:: /../examples/simplest_reset.py
  :language: python

Practical Example 2
-------------------

In this example, we are going to write a reset function for an environment in
which the agent needs to pick up the correctly colored
:py:class:`~gym_gridverse.grid_object.Key` to open a randomly colored
:py:class:`~gym_gridverse.grid_object.Door` and reach the
:py:class:`~gym_gridverse.grid_object.Exit`.

.. literalinclude:: /../examples/choose_key_reset.py
  :language: python
