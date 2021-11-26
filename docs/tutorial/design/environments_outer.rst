Outer Environments
==================

Outer environments are wrappers which contain an inner environment and state
and/or observation representations.  The main  purpose of the outer
environments, is to hide the internal logic of inner environments, which is
based on python object manipulation, and provide an interface based exclusively
on raw numeric data.

.. autoclass:: gym_gridverse.outer_env.OuterEnv
  :noindex:
  :no-members:

An outer environment mostly has the same responsibilities as the corresponding
inner environments, with the main difference being the data format for states
and observations:

* (re)set the initial state (as dictionary of :py:class:`numpy.ndarray`),
* update the state (as dictionary of :py:class:`numpy.ndarray`),
* return the observation (as dictionary of :py:class:`numpy.ndarray`),
* return the reward (as :py:class:`float`),
* return the terminal signal (as :py:class:`bool`).

.. note::
  The conversion between `inner` and `outer` data representations can only be
  performed in one direction:  from `inner` to `outer`.  That means that, while
  it is possible to convert state/observation python objects into raw numeric
  states/observations, it is not possible to convert raw numeric
  states/observations into state/observation objects.  As a consequence, outer
  environments do not provide a functional interface.

Non-Functional Interface
------------------------

The non-functional interface uses the environment's internal state, and
internally refers back to the functional interface by providing that state:

.. automethod:: gym_gridverse.outer_env.OuterEnv.reset
  :noindex:

.. automethod:: gym_gridverse.outer_env.OuterEnv.step
  :noindex:

.. autoattribute:: gym_gridverse.outer_env.OuterEnv.state
  :noindex:

.. autoattribute:: gym_gridverse.outer_env.OuterEnv.observation
  :noindex:

The following figure depicts the inner working of a generic outer environment.

.. figure:: /figures/outer-env-design-dark.png
  :width: 60%
  :align: center
  :figclass: only-dark

  Schematic of the outer environment.

.. figure:: /figures/outer-env-design-light.png
  :width: 60%
  :align: center
  :figclass: only-light

  Schematic of the outer environment.
