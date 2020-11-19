================
Reward Functions
================

The RewardFunction Protocol
===========================

A reward function receives a state, an action, and a next_state, and
deterministically returns a float associated with the given state transition.
Using the :py:mod:`typing` standard library, the reward function type is
defined as

.. autodata:: gym_gridverse.envs.reward_functions.RewardFunction
   :noindex:

Provided Reward Functions
=========================

.. automodule:: gym_gridverse.envs.reward_functions
   :noindex:
   :members: chain,factory
   :undoc-members: RewardFunction
   :show-inheritance:

Custom Reward Functions
=======================

Users can define their own custom reward functions, as long as they satisfy the
following rules:

1. it **MUST** satisfy the
   :py:data:`~gym_gridverse.envs.reward_functions.RewardFunction` protocol.

2. it **MUST** be wholly deterministic.

3. it **MUST NOT** edit the states.

As an example, we are going to write a reward function which returns -1.0 if
the two states in the transition are the same (perhaps this will help the agent
avoid actions which have no effect!)

.. code-block:: python

  from gym_gridverse.state import State
  from gym_gridverse.actions import Actions


  def custom_reward(state: State, action: Actions, next_state: State) -> float:
      """negative reward if state is unchanged"""
      return -1.0 if state == next_state else 0.0

Done!  This reward function can now be used as it is; furthermore, because the
implementation is so generic and task-independent, it can be used with any type
of environment!  Mind you, writing a reward function is not always this easy,
and more complicated reward functions typically have to inspect the inputs to
check if a complicated underlying condition is met;  but regardless of
difficulty, anything is possible!

Parameterization
----------------

Note that we can go one step further and be a bit more fancy with our custom
reward functions.  Let's say that we want to generalize
`custom_reward` such that:

1. it returns a value different than 0.0 if the two states are different

2. it is parametrized such that the


.. code-block:: python

  def custom_reward(
      state: State, action: Actions,
      next_state: State,
      *,
      reward_if_equals: float = -1.0,
      reward_if_not_equals: float = 0.0,
  ) -> float:
      """modulate reward depending on whether state is unchanged"""
      return reward_if_equals if state == next_state else reward_if_not_equals

  custom_reward_stronger = functools.partial(
      custom_reward, reward_if_equals=5.0, reward_if_not_equals=-1.0
  )
  custom_reward_weaker = functools.partial(
      custom_reward, reward_if_equale=0.5, reward_if_not_equals=-0.1
  )


Now you can use functools.partial to set the additional arguments to specific
values, e.g., gotten by command line or some other form of dynamic
configuration, without having to manually editing the code.
