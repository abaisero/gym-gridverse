================
Gym Environments
================

Note that both :py:class:`~gym_gridverse.envs.inner_env.InnerEnv` and
:py:class:`~gym_gridverse.outer_env.OuterEnv` deviate from the OpenAI Gym
interface, which has become a standard in RL research and the associated
software.  This is due to internal design decisions which required additional
and/or separate functionality compared to that of OpenAI Gym.  Nonetheless, we
recognize that many (most?) users will want an interface consistent with that
provided by OpenAI Gym.  For this purpose, we provide two Gym-compliant
wrappers.

GymEnvironment
==============

:py:class:`~gym_gridverse.gym.GymEnvironment` is a wrapper around an `outer`
environment, which makes it compatible with the OpenAI Gym interface.  This is
the type of environment returned by :py:func:`gym.make` if one of the GV
predefined environment ids is used.  Compared to a standard OpenAI Gym
environment, :py:class:`~gym_gridverse.gym.GymEnvironment` provides additional
state information via attributes
:py:attr:`~gym_gridverse.gym.GymEnvironment.state_space` and
:py:attr:`~gym_gridverse.gym.GymEnvironment.state`.

GymStateWrapper
===============

Despite providing state information,
:py:class:`~gym_gridverse.gym.GymEnvironment` is still meant to represent
partially observable control problem, which is reflected by the fact that the
:py:meth:`~gym_gridverse.gym.GymEnvironment.reset` and
:py:meth:`~gym_gridverse.gym.GymEnvironment.step` methods return the
observation representations.  :py:class:`~gym_gridverse.gym.GymStateWrapper`
changes the :py:meth:`~gym_gridverse.gym.GymStateWrapper.reset` and
:py:meth:`~gym_gridverse.gym.GymStateWrapper.step` methods to return the state
representations, and thus truly represents underlying fully observable version
of the control problem.
