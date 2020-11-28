====================
Visibility Functions
====================

In this section we describe the visibility function protocol, the visibility
functions provided in :py:mod:`~gym_gridverse.envs.visibility_functions`, and
how to write your own custom visibility functions.

The VisibilityFunction Protocol
===============================

A visibility function is a generative function which represents a (stochastic)
distribution over binary visibility grids.  Using the :py:mod:`typing` standard
library, the visibility function type is defined as a
:py:class:`typing.Protocol` with a :py:meth:`__call__` member which receives a
:py:class:`~gym_gridverse.info.Grid` representing the agent's restricted view,
a :py:class:`~gym_gridverse.geometry.Position` representing the agent's
position in the grid, and an optional :py:class:`numpy.random.Generator`, and
returns a :py:class:`numpy.ndarray` indicating whether each tile is visible or
not.

.. autoclass:: gym_gridverse.envs.visibility_functions.VisibilityFunction
    :noindex:
    :members: __call__

.. note::
    A visibility function may (and often does) accept additional arguments;  this
    is possible **so long as** the extra arguments either have default values,
    or are binded to specific values later on, e.g., using
    :py:func:`functools.partial`.

Provided Visibility Functions
=============================

.. warning::
    TODO

Custom Visibility Functions
===========================

Custom visibility functions can be defined so long as they satisfy some basic
rules;  A custom visibility function:

- **MUST** satisfy the
  :py:class:`~gym_gridverse.envs.visibility_functions.VisibilityFunction`
  protocol.

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

Practical Example 1
-------------------

In this example, we are going to write a rather questionable visibility
function in which the visibility of every other tile is determined by a coin
flip (ignoring tile transparency).

.. code-block:: python

    from gym_gridverse.info import Grid
    from gym_gridverse.geometry import Position
    import numpy.random as rnd


    def coinflip_visibility(
        grid: Grid,
        position: Position,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> np.ndarray:
        """randomly determines tile visibility"""
    
        rng = get_rng_if_none(rng)
    
        visibility = rng.integers(2, size=grid.shape).astype(bool)
        visibility[position.y, position.x] = True  # agent always visible
    
        return visibility

Practical Example 2
-------------------

In this example, we are going to write an objectionable visibility function in
which the agent's field of view expands like a 90° angle cone (ignoring tile
transparency).

.. code-block:: python

    from gym_gridverse.info import Grid
    from gym_gridverse.geometry import Position
    import numpy.random as rnd


    def conic_visibility(
        grid: Grid,
        position: Position,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> np.ndarray:
        """cone-shaped visibility, passes through objects"""

        visibility = np.zeros(grid.shape, dtype=bool)

        for y in range(position.y, -1, -1):
            dy = position.y - y

            x_from = position.x - dy
            x_to = position.x + dy
            for x in range(x_from, x_to + 1):
                visibility[y, x] = True
    
        return visibility
