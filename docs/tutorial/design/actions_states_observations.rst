Actions, States, and Observations
=================================

Before seeing how the environments glue everything together, let us have a
brief look into how actions, states, and observations are represented by their
respective classes.

.. autoclass:: gym_gridverse.action.Action
  :noindex:

.. autoclass:: gym_gridverse.state.State
  :noindex:

.. autoclass:: gym_gridverse.observation.Observation
  :noindex:

States and observations are very simple container classes, and the details of
what they contain are all in the respective
:py:class:`~gym_gridverse.grid.Grid` and :py:class:`~gym_gridverse.agent.Agent`
fields.  However, for the time being, we will postpone looking into those
classes, and move onto a very important design choice:  the inner-outer
environment separation.
