=============
Customization
=============

In this section, we will learn more about the components which make a
:py:class:`~gym_gridverse.envs.gridworld.GridWorld`, i.e.,
:py:class:`~gym_gridverse.envs.reset_functions.ResetFunction`,
:py:class:`~gym_gridverse.envs.transition_functions.TransitionFunction`,
:py:class:`~gym_gridverse.envs.reward_functions.RewardFunction`,
:py:class:`~gym_gridverse.envs.observation_functions.ObservationFunction`,
:py:class:`~gym_gridverse.envs.visibility_functions.VisibilityFunction`,
:py:class:`~gym_gridverse.envs.terminating_functions.TerminatingFunction`.
Finally, we will also learn how to create a custom
:py:class:`~gym_gridverse.grid_object.GridObject`.  In each of the following
pages, we will learn in detail about the respective protocols, registries,
customization requirements, and some practical examples.

.. toctree::
  :hidden:
  :maxdepth: 2

  customization/reset_functions
  customization/transition_functions
  customization/reward_functions
  customization/observation_functions
  customization/visibility_functions
  customization/terminating_functions
  customization/grid_object
  customization/yaml
