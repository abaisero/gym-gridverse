====
Grid
====

A :py:class:`~gym_gridverse.grid.Grid` represents the contents of the gridworld
itself, i.e., the objects in the grid (instances of
:py:class:`~gym_gridverse.grid_object.GridObject`), their properties, and the
spatial relationships between them.  Note, however, that this **excludes** any
information about the agent itself, which is stored in a separate
:py:class:`~gym_gridverse.agent.Agent` instance.  The
:py:class:`~gym_gridverse.grid.Grid` class provides an interface to get, set,
and change its contents.

.. autoclass:: gym_gridverse.grid.Grid
  :noindex:

.. note::

  Cells of a grid are indexed using a combination of matrix-like indexing
  convention and some pythong indexing conventions:

  .. figure:: /figures/grid.png
    :align: center

    Grid

  .. figure:: /figures/grid-coordinates.png
    :align: center

    Indexing

  * The first axis is ``y``, which expands `downwards` (as in matrix indexing),
  * The second axis is ``x``, which expands `rightwards` (as in matrix indexing),
  * Indexing starts at :math:`(0,0)`, which represents the top-left cell (as in python indexing),
  * Negative indices wrap around once, i.e., :math:`(-1,-1)` represents the bottom-right cell (as in python indexing),
  * If the grid has shape :math:`(h,w)`, indices outside of the ranges :math:`[-h,h-1]` and :math:`[-w,w-1]` results in an ``IndexError`` being raised (as in python indexing).
