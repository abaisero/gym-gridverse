# Gym Gridverse
Gridworld domains in the gym interface

# Specification

## Env API

* Gym
* Functional step (give/return state)
* `get_used_objects` (for representing states/observations more compactly)

## Grid

* Int representation
* Global mapping object type -> int

## Domain specifications

* Gym interface (minimal)
* Grid (2-dimensional)
* Partially observable
* Spaces (discrete)
    * States: [ height x width x channels ]
    * Observations: [ height x width x channels ]
        * agent POV
    * Action: [ n-agents ]
* State is captured by grid
    * No notion of state outside of grid
    * Objects do not stack

## Grid & Object representation

* Grid: [height x width] objects
* Translation to [height x width x channels] ???
    * Channel 1 supposedly object type
* Each domain has a personal mapping from object type -> int
* Object status?

# Roadmap

1. Design decision
    * How to represent state in integer form across domains
    * Factorize domain logic (give functions vs complex default behavior)
