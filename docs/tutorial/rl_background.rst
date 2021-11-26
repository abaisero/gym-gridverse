=============
RL Background
=============

Here, we define the types of control problems which GV is meant for your to
construct.  Although GV is specifically meant for grid-world control problems,
the following definitions are more generic.

Markov Decision Processes (MDPs)
================================

An MDP is a discrete time control problem composed from the following components:

* A state space :math:`\mathcal{S}`,
* An action space :math:`\mathcal{A}`,
* A state transition function :math:`\mathcal{T}\colon\mathcal{S}\times\mathcal{A}\to\Delta\mathcal{S}`, which represents the probability distribution of the next state given the previous state and action :math:`\Pr(s_{t+1}\mid s_t, a_t)`,
* A reward function :math:`R\colon \mathcal{S}\times\mathcal{A}\times\mathcal{S}\to\mathbb{R}`,
* A discount value :math:`\gamma\in [0,1]`.

An MDP agent is represented by a policy function
:math:`\pi\colon\mathcal{S}\to\Delta\mathcal{A}`, which represents how actions
are chosen from given states.  The goal of the control problem is that to
compute (or otherwise determine) the optimal policy which maximizes the total
return

.. math::
  \pi^* = \arg\max_\pi \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t, s_{t+1}) \right] \,.

Partially Observable MDPs (POMDPs)
==================================

A POMDP is a discrete time control problem composed from the following
components:

* A state space :math:`\mathcal{S}`,
* An observation space :math:`\mathcal{O}`,
* An action space :math:`\mathcal{A}`,
* A state transition function :math:`\mathcal{T}\colon\mathcal{S}\times\mathcal{A}\to\Delta\mathcal{S}`, which represents the probability distribution of the next state given the previous state and action :math:`\Pr(s_{t+1}\mid s_t, a_t)`,
* An observation function function :math:`\mathcal{O}\colon\mathcal{S}\times\mathcal{A}\times{S}\to\Delta\mathcal{S}`, which represents the probability distribution of an observation given the state transition which just cocurred :math:`\Pr(o_{t+1}\mid s_t, a_t, s_{t+1})`,
* A reward function :math:`R\colon \mathcal{S}\times\mathcal{A}\times\mathcal{S}\to\mathbb{R}`,
* A discount value :math:`\gamma\in [0,1]`.

In a POMDP, the agent is not able to observe the underlying system state
directly, but rather only observes an indirect observation which contains
partial information about the real state of the environment.  For this reason,
the agent cannot select actions based on the state, but rather should select
actions based on the entire observable past, a.k.a. the history, which is
composed of all previous actions and observations.  Therefore, we introduce the
the histroy space
:math:`\mathcal{H}=\left(\mathcal{A}\times\mathcal{O}\right)^*`.  A POMDP agent
is represented by a policy function
:math:`\pi\colon\mathcal{H}\to\Delta\mathcal{A}`, and the goal is that to
compute (or otherwise determine) the optimal policy which maximizes the total
return

.. math::
  \pi^* = \arg\max_\pi \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t, s_{t+1}) \right] \,.

GV Assumptions
==============

Compared to generic forms of MDPs and PODMPs, GV makes further assumptions:

* The state space is formed by a grid of cells which contains objects with properties, combined with an agent which may move and turn around, actuate, and pick up objects.
* The observation space is similar to the state space, albeit representing the agent's POV (including limited view range and occlusions).
* The observation function only depends on the current state, rather than the entire state-action-state transition,
* GV does not handle the discount factor :math:`\gamma`;  user code needs to determine the discount factor on its own and use it appropriately.
