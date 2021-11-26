=====
Agent
=====

An :py:class:`~gym_gridverse.agent.Agent` is a simple container object which
represents information about the agent, i.e., its position and orientation, and
the :py:class:`~gym_gridverse.grid_object.GridObject` which it holds.  Note
that, despite its name, the :py:class:`~gym_gridverse.agent.Agent` object does
not represent the decision process of the agent, but only information relative
to the agent in the environment.

.. autoclass:: gym_gridverse.agent.Agent
  :noindex:

.. note::
  
  :py:class:`~gym_gridverse.agent.Agent` objects are used in both
  :py:class:`~gym_gridverse.state.State` and
  :py:class:`~gym_gridverse.observation.Observation`, and thus contextually
  contains different types of information.  For example, the
  :py:class:`~gym_gridverse.agent.Agent` instance of an
  :py:class:`~gym_gridverse.observation.Observation` object may not contain the
  agent's **global** state position, but only its local position (relative to
  its POV).
