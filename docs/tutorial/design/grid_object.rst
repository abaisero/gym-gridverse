==========
GridObject
==========

A :py:class:`~gym_gridverse.grid_object.GridObject` represents an object in the
gridworld, its properties, and partially its behavior.  A
:py:class:`~gym_gridverse.grid_object.GridObject` requires certain attributes
and methods to be defined.

.. autoclass:: gym_gridverse.grid_object.GridObject
  :noindex:

.. note::

  There are two special types of grid objects which are only used in special
  occasions:

  NoneGridObject
    A :py:class:`~gym_gridverse.grid_object.NoneGridObject` is the object held
    by the agent by default if it is not holding any other
    gym_gridverse.grid_object.GridObject.

  Hidden
    A :py:class:`~gym_gridverse.grid_object.Hidden` is the grid object used to
    indicate a non-observable 

Class Methods
=============

A :py:class:`~gym_gridverse.grid_object.GridObject` class provides the
following class methods:

.. automethod:: gym_gridverse.grid_object.GridObject.num_states
  :noindex:

.. automethod:: gym_gridverse.grid_object.GridObject.can_be_represented_in_state
  :noindex:


Instance Attributes
===================

A :py:class:`~gym_gridverse.grid_object.GridObject` has the following attributes:

.. autoattribute:: gym_gridverse.grid_object.GridObject.color
  :noindex:

.. autoattribute:: gym_gridverse.grid_object.GridObject.state_index
  :noindex:

.. autoattribute:: gym_gridverse.grid_object.GridObject.blocks_vision
  :noindex:

.. autoattribute:: gym_gridverse.grid_object.GridObject.blocks_movement
  :noindex:

.. autoattribute:: gym_gridverse.grid_object.GridObject.holdable
  :noindex:

