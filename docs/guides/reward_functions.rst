================
Reward Functions
================

In this section we describe the reward function protocol, the reward functions
provided in :py:mod:`~gym_gridverse.envs.reward_functions`, and how to write
your own custom reward functions.

The RewardFunction Protocol
===========================

A reward function is a deterministic mapping from a state-action-state
transition to a numeric reward.  Using the :py:mod:`typing` standard library,
the reward function type is defined as a :py:class:`typing.Callable` which
receives a :py:class:`~gym_gridverse.state.State`, an
:py:class:`~gym_gridverse.actions.Actions`, and another
:py:class:`~gym_gridverse.state.State`, and returns a :py:class:`float`.

.. autodata:: gym_gridverse.envs.reward_functions.RewardFunction
   :noindex:

.. note::
  A reward function may (and often does) accept additional arguments;  this is
  possible **so long as** the extra arguments either have default values, or
  are binded to specific values later on, e.g., using
  :py:func:`functools.partial`.

Provided Reward Functions
=========================

The :py:mod:`~gym_gridverse.envs.reward_functions` module contains some
predefined reward functions, among which:

- :py:func:`~gym_gridverse.envs.reward_functions.living_reward` -- a
  constant reward.

- :py:func:`~gym_gridverse.envs.reward_functions.overlap` -- a reward for being
  on the same tile as a unique
  :py:class:`~gym_gridverse.grid_object.GridObject`.

- :py:func:`~gym_gridverse.envs.reward_functions.reach_goal` -- a reward for
  reaching the :py:class:`~gym_gridverse.grid_object.Goal`.

- :py:func:`~gym_gridverse.envs.reward_functions.getting_closer` -- a reward
  for moving closer to a unique
  :py:class:`~gym_gridverse.grid_object.GridObject`.

- :py:func:`~gym_gridverse.envs.reward_functions.proportional_to_distance` -- a
  reward based on the distance to a unique
  :py:class:`~gym_gridverse.grid_object.GridObject`.

.. note::
  Reward functions are modular and can be combined to costruct more complicated
  or specialized rewards by calling each other, e.g., the implementation of
  :py:func:`~gym_gridverse.envs.reward_functions.reach_goal` internally refers
  to :py:func:`~gym_gridverse.envs.reward_functions.overlap`.  A standard way
  to combine multiple reward functions is using
  :py:func:`~gym_gridverse.envs.reward_functions.chain`, which returns the sum
  of rewards determined by other reward functions.

Custom Reward Functions
=======================

Custom reward functions can be defined so long as they satisfy some basic rules;  A
custom reward function:

- **MUST** satisfy the
  :py:data:`~gym_gridverse.envs.reward_functions.RewardFunction` protocol.

- **MUST NOT** edit the input states.

- **SHOULD** be wholly deterministic.

.. warning::
    The reward function is usually deterministic by the definition of the
    reinforcement learning control problem.  Regardless, introducing reward
    stochasticity does not fundamentally change the nature of a control
    problem, but rather "only" makes the agent feedback noisier.
    
    While we discourage implementing and using stochastic reward functions, you
    may use them if you wish to.  The primary drawback will be that seeding the
    environment using :py:meth:`~gym_gridverse.envs.env.Environment.set_seed`
    will be insufficient to reproduce traces, runs, and experiments;  if you
    wish to maintain reproducibility despite employing a stochastic reward
    function, you will have to manage the external source of randomness and its
    seeding yourself.

Practical Example 1
-------------------

In this example, we are going to write a reward function which returns -1.0 if
the two states in the transition are the same (perhaps this will help the agent
avoid actions which have no effect!).

.. literalinclude:: example__reward_function__static_reward.py
  :language: python
  :caption: example static reward function
  :name: static_reward

Done! This reward function can now be used as it is; furthermore, because the
implementation is so generic and task-independent, it can be used with any type
of environment!  Mind you, writing a reward function is not always this easy,
and more complicated reward functions typically have to inspect the inputs to
check if a complicated underlying condition is met (which is the subject of the
second example in this guide);  but regardless of difficulty, anything is
possible!

We can go one step further and generalize :ref:`static_reward` such that
different reward values can be used without having to manually edit the code
each time.  We do this by adding appropriate arguments to the function
signature, and then using :py:func:`functools.partial` with values which might
come from command line arguments or a file configuration.

.. literalinclude:: example__reward_function__generalized_static_reward.py
  :language: python

Practical Example 2
-------------------

In this example, we are going to write a reward which encourages the agent to
turn around in a given direction, clockwise or counterclockwise (perhaps this
will help the agent gain more information by changing POV!)  We provide two
different implementations which will hopefully demonstrate the intricacies of
coding the correct functionality.

First Implementation
^^^^^^^^^^^^^^^^^^^^
In the first implementation, we will simply check whether the ``action``
argument matches one of the rotation actions
(:py:attr:`~gym_gridverse.actions.Actions.TURN_LEFT` and
:py:attr:`~gym_gridverse.actions.Actions.TURN_RIGHT`), and select the
appropriate reward:

.. literalinclude:: example__reward_function__intended_rotation_reward.py
  :language: python

Easy!  Note, however, that we did not use the ``state`` and
``next_state`` arguments at all;  should we be worried about that?  As
it turns out, this implementation measures the agent's *intention* to turn, but
not necessarily whether the agent *actually* turned.  The two conditions might
(or might not) be very different, depending on the state dynamics:

- the agent might have tried to turn but failed, e.g., due to a flat 10\%
  failure rate in actually performing actions.

- the agent might have turned by means other than its action, e.g., by standing
  on a rotating tile.

Second Implementation
^^^^^^^^^^^^^^^^^^^^^

If we wanted to re-implement this reward function by taking into account what
*actually* happened in the transition, we might do it as follows:

.. literalinclude:: example__reward_function__actual_rotation_reward.py
  :language: python

It is up to you, the designer, to know your environment well enough not only to
decide what kind of behavior to reward, but also to be able to encode the
concept of a reward into an implementation which correctly matches that
concept.
