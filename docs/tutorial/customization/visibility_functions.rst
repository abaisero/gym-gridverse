====================
Visibility Functions
====================

In this section we describe the visibility function protocol, the visibility
function registry, and how to write your own custom visibility functions.

The VisibilityFunction Protocol
===============================

A visibility function is a generative function which represents a (stochastic)
distribution over binary visibility grids.  Using the :py:mod:`typing` standard
library, the visibility function type is defined as a callable
:py:class:`typing.Protocol` which receives a
:py:class:`~gym_gridverse.grid.Grid` representing the agent's restricted view,
a :py:class:`~gym_gridverse.geometry.Position` representing the agent's
position in the grid, and an optional :py:class:`numpy.random.Generator`, and
returns a boolean :py:class:`numpy.ndarray` indicating whether each tile is
visible or not.

.. autoclass:: gym_gridverse.envs.visibility_functions.VisibilityFunction
    :noindex:
    :members: __call__

.. note::
  A visibility function may accept additional arguments;  this is possible **so
  long as** the extra arguments either have default values, or are binded to
  specific values later on, e.g., using :py:func:`functools.partial`.

The Visibility Function Registry
================================

The :py:mod:`~gym_gridverse.envs.visibility_functions` module contains some
pre-defined visibility functions and the
:py:data:`~gym_gridverse.envs.visibility_functions.visibility_function_registry`,
a dictionary-like object through which to register and retrieve visibility
functions.  Visibility functions are registered using the
:py:meth:`~gym_gridverse.utils.registry.FunctionRegistry.register` method,
which can be used as a decorator (also see
:ref:`tutorial/customization/visibility_functions:Custom Visibility
Functions`).  As a dictionary,
:py:data:`~gym_gridverse.envs.visibility_functions.visibility_function_registry`
has a :py:meth:`~dict.keys` method which returns the names of registered
functions.

Custom Visibility Functions
===========================

Custom visibility functions can be defined so long as they satisfy some basic
rules;  A custom visibility function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.visibility_functions.VisibilityFunction`
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

Practical Example 1
-------------------

.. note::
  The examples shown here can be found in the ``examples/`` folder.

In this example, we are going to write a rather questionable visibility
function in which the visibility of every other tile is determined by a coin
flip (ignoring tile transparency).

.. literalinclude:: /../examples/coinflip_visibility.py
  :language: python

Practical Example 2
-------------------

In this example, we are going to write a visibility function in which the
agent's field of view expands like a 90° angle cone (ignoring tile
transparency).

.. literalinclude:: /../examples/conic_visibility.py
  :language: python
