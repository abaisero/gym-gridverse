Representations
===============

In GV, states and observations are represented by python objects.  In order to
process them using neural network models, we need a way to convert them into
raw numeric form.  In GV, this process is performed by the
:py:class:`~gym_gridverse.representations.representation.Representation`
classes.  A representation converts state/observation spaces into numeric
spaces (as dictionaries of
:py:class:`~gym_gridverse.representations.spaces.Space`), and
:py:class:`~gym_gridverse.state.State`/:py:class:`~gym_gridverse.observation.Observation`
into numeric states and observations (as dictionaries of numpy arrays
(:py:class:`~numpy.ndarray`)

State Representations
---------------------

Available state representations can be found in
:py:mod:`~gym_gridverse.representations.state_representations`.  State
representations must implement the abstract interface
:py:class:`~gym_gridverse.representations.representation.StateRepresentation`,
which requires the implementation of a
:py:meth:`~gym_gridverse.representations.representation.Representation.space`
property and a
:py:meth:`~gym_gridverse.representations.representation.StateRepresentation.convert`
method:

.. autoclass:: gym_gridverse.representations.representation.StateRepresentation
  :noindex:
  :members: space, convert

To create one of the representations defined in the GV library, you can use the respective function:

.. autofunction:: gym_gridverse.representations.state_representations.make_state_representation
   :noindex:

Observation Representations
---------------------------

Available observation representations can be found in
:py:mod:`~gym_gridverse.representations.observation_representations`.
Observation representations must implement the abstract interface
:py:class:`~gym_gridverse.representations.representation.ObservationRepresentation`,
which requires the implementation of a
:py:meth:`~gym_gridverse.representations.representation.Representation.space`
property and a
:py:meth:`~gym_gridverse.representations.representation.ObservationRepresentation.convert`

.. autoclass:: gym_gridverse.representations.representation.ObservationRepresentation
  :noindex:
  :members: space, convert

To create one of the representations defined in the GV library, you can use the respective function:

.. autofunction:: gym_gridverse.representations.observation_representations.make_observation_representation
   :noindex:

