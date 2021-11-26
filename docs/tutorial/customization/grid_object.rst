===========
GridObjects
===========

In this section we describe the grid-object protocol, the grid-object registry,
and how to write your own custom grid-objects.

The GridObject Protocol
=======================

A grid-object is a type of python class which represents the contents of a
:py:class:`~gym_gridverse.grid.Grid`.  A grid-object needs to inherit from
:py:class:`~gym_gridverse.grid_object.GridObject` and define some required
attributes, properties, and classmethods.

.. autoclass:: gym_gridverse.grid_object.GridObject
  :noindex:
  :members: state_index, color, blocks_movement, blocks_vision, holdable, can_be_represented_in_state, num_states

The GridObject Registry
=======================

The :py:mod:`~gym_gridverse.grid_object` module contains some pre-defined
grid-objects and the
:py:data:`~gym_gridverse.grid_object.grid_object_registry`, a list-like object
through which to register and retrieve grid-objects.  Grid-objects are
automatically registered through inheritance from
:py:class:`~gym_gridverse.grid_object.GridObject`.
:py:data:`~gym_gridverse.grid_object.grid_object_registry` has a
:py:meth:`~gym_gridverse.grid_object.GridObjectRegistry.names` method which
returns the names of registered grid-object classes, and a
:py:meth:`~gym_gridverse.grid_object.GridObjectRegistry.from_name` method which
returns the grid-object class associated with a name.

Custom GridObjects
==================

Custom grid-objects can be defined so long as they satisfy some basic rules;  A
custom grid-object:

- **MUST** define the
  :py:data:`~gym_gridverse.grid_object.GridObject.state_index`,
  :py:data:`~gym_gridverse.grid_object.GridObject.color`,
  :py:data:`~gym_gridverse.grid_object.GridObject.blocks_movement`,
  :py:data:`~gym_gridverse.grid_object.GridObject.blocks_vision`, and
  :py:data:`~gym_gridverse.grid_object.GridObject.holdable` attributes as
  either class attributes, instance attributes, or instance properties.

- **MUST** implement the
  :py:meth:`~gym_gridverse.grid_object.GridObject.num_states` and
  :py:meth:`~gym_gridverse.grid_object.GridObject.can_be_represented_in_state`
  class methods.

.. tip::

  We advise implementing each of the
  :py:data:`~gym_gridverse.grid_object.GridObject.state_index`,
  :py:data:`~gym_gridverse.grid_object.GridObject.color`,
  :py:data:`~gym_gridverse.grid_object.GridObject.blocks_movement`,
  :py:data:`~gym_gridverse.grid_object.GridObject.blocks_vision`, and
  :py:data:`~gym_gridverse.grid_object.GridObject.holdable` attributes as:

  * **class attributes** if all instances of the same grid-object class should have
    the same constant value, e.g.,
    :py:data:`~gym_gridverse.grid_object.Door.holdable` is implemented as a
    class attribute because doors are never holdable.

  * **instance attributes** if different instances of the same grid-object class
    can have different values, but the value associated with each instances is
    constant and does not change, e.g.,
    :py:data:`~gym_gridverse.grid_object.Door.color` is implemented as an
    instance attribute because doors can have different colors, but they don't
    change colors.

  * **properties** if different instances of the same grid-boject class can
    have different values, and the value associated with each instance can also
    vary, e.g., :py:data:`~gym_gridverse.grid_object.Door.state_index`,
    :py:data:`~gym_gridverse.grid_object.Door.blocks_movement`, and
    :py:data:`~gym_gridverse.grid_object.Door.blocks_vision` are impemented as
    properties because they will depend on whether the door is open or closed,
    which is determined by a separate
    :py:data:`~gym_gridverse.grid_object.Door.state` attribute.

.. attention::

  If any of the grid-object attributes are implemented as instance attributes,
  you'll need to overwrite the respective
  :py:class:`~gym_gridverse.grid_object.GridObject` abstract property.  This
  can be done by simply specifying a class-level typehint::

    class SomeColoredGridObject(GridObject):
      color: Color  # this typehint is required to make `color` an instance attribute

      def __init__(self, color: Color):
        self.color = color
        ...  # rest of initialization

      ...  # rest of class definition

Practical Examples 1 & 2
------------------------

.. note::

  The examples shown here can be found in the ``examples/`` folder.

In the followinng examples, we are going to create two grid-objects which
thematically influence the agent movement.  The first grid-objoect is an
``Ice``, which may have two statuses:  ``SMOOTH``, which indicates that the
agent may walk on it, and ``BROKEN`` wich indicates that the agent may not walk
on it.  Further, it may be thematically interesting to make the agent slip when
moving on a ``SMOOTH`` ``Ice``, continuing its movement until it hits an
obstacle.

.. important::

  Note that the grid-object only defines these properties and attributes of an
  ``Ice``, while the behavior of the agent trying to walk on the tile should be
  implemented separately as a
  :py:class:`~gym_gridverse.envs.transition_functions.TransitionFunction` (we
  leave this as an exercise).

.. literalinclude:: /../examples/ice_grid_object.py
  :language: python

The second grid-object is ``IceCleats``, a holdable object which may be used to
help an agent walk on a ``SMOOTH`` ``Ice`` without slipping.  Once again, the
behavior of an agent walking on ``Ice`` wihle holding ``IceCleats`` should be
implemented by an appropriate
:py:class:`~gym_gridverse.envs.transition_functions.TransitionFunction`.

.. literalinclude:: /../examples/ice_cleats_grid_object.py
  :language: python
