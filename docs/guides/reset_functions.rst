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

.. warning::
    TODO

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

.. warning::
    The :py:data:`rng` argument is to control the source of randomness and
    allow for the environment to be seeded via
    :py:meth:`~gym_gridverse.envs.env.Environment.set_seed`, which in turn
    guarantees the reproducibility of traces, runs, and experiments;  if you
    wish to use external sources of randomness, you will have to manage them
    and their seeding yourself.

.. warning::
    TODO
