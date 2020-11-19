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

.. warning::
    TODO

Custom Observation Functions
============================

.. warning::
    TODO
