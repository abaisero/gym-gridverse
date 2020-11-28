====================
Transition Functions
====================

In this section we describe the transition function protocol, the transition
functions provided in :py:mod:`~gym_gridverse.envs.state_dynamics`, and
how to write your own custom transition functions.

The TransitionFunction Protocol
===============================

A transition function is a generative function which represents a (stochastic)
mapping from a state-action pair to a next states.  Using the :py:mod:`typing`
standard library, the transition function type is defined as a
:py:class:`typing.Protocol` with a :py:meth:`__call__` member which receives a
:py:class:`~gym_gridverse.state.State`, an
:py:class:`gym_gridverse.actions.Actions`, and an optional
:py:class:`numpy.random.Generator`, and edits the input
:py:class:`~gym_gridverse.state.State`.

.. autoclass:: gym_gridverse.envs.transition_functions.TransitionFunction
    :noindex:
    :members: __call__

.. note::
    A transition function may (and often does) accept additional arguments;
    this is possible **so long as** the extra arguments either have default
    values, or are binded to specific values later on, e.g., using
    :py:func:`functools.partial`.

Provided Transition Functions
=============================

.. warning::
    TODO

Custom Transition Functions
===========================

Custom transition functions can be defined so long as they satisfy some basic
rules;  A custom transition function:

- **MUST** satisfy the
  :py:class:`~gym_gridverse.envs.transition_functions.TransitionFunction`
  protocol.

- **MUST** edit the input state, rather than return a new state altogether.

- **SHOULD** use the :py:data:`rng` argument as the source for any
  stochasticity.

- **MUST** use :py:func:`~gym_gridverse.rng.get_gv_rng_if_none` (only if the
  :py:data:`rng` is used at all).

.. warning::
    The :py:data:`rng` argument is to control the source of randomness and
    allow for the environment to be seeded via
    :py:meth:`~gym_gridverse.envs.env.Environment.set_seed`, which in turn
    guarantees the reproducibility of traces, runs, and experiments;  if you
    wish to use external sources of randomness, you will have to manage them
    and their seeding yourself.

Practical Example 1
-------------------

In this example, ...

.. warning::
    TODO

.. code-block:: python

    from gym_gridverse.state import State
    from gym_gridverse.actions import Actions
    from gym_gridverse.rng import get_rng_if_none
    from gym_gridverse.grid_object import Floor, Wall


    def creeping_walls(
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        """randomly chooses a Floor tile and turns it into a Wall tile"""
    
        from gym_gridverse.grid_object import Floor, Wall
    
        rng = get_rng_if_none(rng)  # necessary to use rng object!
    
        floor_positions = [
            position
            for position in state.grid.positions()
            if isinstance(state.grid[position], Floor)
        ]
    
        try:
            position = rng.choice(floor_positions)
        except ValueError:  # no floor positions
            pass
        else:
            state.grid[position] = Wall()

Practical Example 2
-------------------

In this example, we are going to write a transition function which randomizes
the execution of another transition function.

.. code-block:: python

    from gym_gridverse.state import State
    from gym_gridverse.actions import Actions
    from gym_gridverse.rng import get_rng_if_none


    def random_transition(
        state: State,
        action: Actions,
        *,
        transition_function: TransitionFunction,
        p_success: float,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        """randomly determines whether to perform a transition or not"""

        rng = get_rng_if_none(rng)  # necessary to use rng object!
        
        success = rng.random() <= p_success
        if success:
            transition_function(state, action, rng=rng)
