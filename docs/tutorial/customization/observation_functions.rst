=====================
Observation Functions
=====================

In this section we describe the observation function protocol, the observation
function registry, and how to write your own custom observation functions.

The ObservationFunction Protocol
================================

An observation function is a generative function which represents a
(stochastic) mapping from a state to an observation.  Using the
:py:mod:`typing` standard library, the observation function type is defined as
a callable :py:class:`typing.Protocol` which receives a
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

The Observation Function Registry
=================================

The :py:mod:`~gym_gridverse.envs.observation_functions` module contains some
pre-defined observation functions and the
:py:data:`~gym_gridverse.envs.observation_functions.observation_function_registry`,
a dictionary-like object through which to register and retrieve observation
functions.  Observation functions are registered using the
:py:meth:`~gym_gridverse.utils.registry.FunctionRegistry.register` method,
which can be used as a decorator (also see
:ref:`tutorial/customization/observation_functions:Custom Observation
Functions`).  As a dictionary,
:py:data:`~gym_gridverse.envs.observation_functions.observation_function_registry`
has a :py:meth:`~dict.keys` method which returns the names of registered
functions.

Custom Observation Functions
============================

.. note::
  While writing your own observation function directly is indeed a possibility,
  the most common way to implement new observation functions is by writing a
  custom :py:data:`~gym_gridverse.envs.visibility_functions.VisibilityFunction`
  and using it with the
  :py:func:`~gym_gridverse.envs.observation_functions.from_visibility`
  observation function.

Custom observation functions can be defined so long as they satisfy some basic
rules;  A custom observation function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.observation_functions.ObservationFunction`
  protocol.

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

Practical Example
-----------------

.. note::
  The examples shown here can be found in the ``examples/`` folder.

In this example, we are going to write an observation function which simulates
a satellite view of the agent, i.e., a view from the top (being able to see
through walls), and with the agent being off-center in each observation.

.. literalinclude:: /../examples/satellite_observation.py
  :language: python
