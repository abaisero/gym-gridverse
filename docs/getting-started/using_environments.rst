==================
Using Environments
==================

.. todo::

  - Finish

You will typically interact with our package through the
:py:class:`~gym_gridverse.outer_env.OuterEnv`, which bundles together the actual
:py:class:`~gym_gridverse.envs.inner_env.InnerEnv` with a
:py:class:`~gym_gridverse.representations.representation.StateRepresentation`
and
:py:class:`~gym_gridverse.representations.representation.ObservationRepresentation`.

.. autoclass:: gym_gridverse.outer_env.OuterEnv
  :noindex:
  :members:

Example scripts
===============

A script that prints the interactions of a random agent to the terminal
(:download:`scripts/visualize_random_bot_in_terminal.py
<../../scripts/visualize_random_bot_in_terminal.py>`):

.. literalinclude:: ../../scripts/visualize_random_bot_in_terminal.py
  :language: python
