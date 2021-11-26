==================
Using Environments
==================

In the following sections, we show simple random-agent control loops which
respectively use GV's "outer" interface, and the OpenAI gym interface.

Using the GV "Outer" interface
==============================

The "outer" interface will be explained in detail in the
:ref:`tutorial/design:Design` section of this tutorial.  For now, it will
suffice to know that it provides an alternative interface to interact with an
environment.

This script is also available as :download:`scripts/gv_control_loop_outer.py
</../scripts/gv_control_loop_outer.py>`.

.. literalinclude:: /../scripts/gv_control_loop_outer.py
  :language: python

Using the OpenAI Gym interface
==============================

The OpenAI Gym interface is implemented by
:py:class:`~gym_gridverse.gym.GymEnvironment`.  In addition to the fields
defined by the gym interface itself, this class provides access to
:py:attr:`~gym_gridverse.gym.GymEnvironment.state_space` and
:py:attr:`~gym_gridverse.gym.GymEnvironment.state` attributes.

This script is also available as :download:`scripts/gv_control_loop_gym.py
</../scripts/gv_control_loop_gym.py>`.

.. literalinclude:: /../scripts/gv_control_loop_gym.py
  :language: python
