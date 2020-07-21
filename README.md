# Gym Gridverse

Gridworld domains in the gym interface

# Road map

1. BlahBlah

# Specification

## Env API

* Gym
* Functional step (give/return state)
* `get_used_objects` (for representing states/observations more compactly)

## Action:

* turn left/right 
* move left/right/forward/backward
* pick up & down
* actuate

## State:

* Require
    * Int representation
    * Global mapping object type -> int
    * Per object: mapping properties -> number
* Representation:
    * Grid: W x H x 4 channels
        * Numpy array of objects
        * Rendered:
            1. bit for agent representation
            1. cell type
            1. cell status
            1. cell color
    * non-spatial features 
        * contains:
            * Direction
            * Type of holding object
            * Property 1 of holding object
            * Property 2 of holding object
        * Renderable into tensor

## Observation

* [slice grid, state feature]
* Encoding of hidden cells
* Agent POV

## Wrappers

* Map cell status properties to unique number per type
* One-hot encoding
* Compacting: mapping from global object type number -> domain specific
* Maps actions to original Minigrid actions

## Domain specifications

* Gym interface (minimal)
* Grid (2-dimensional)
* Partially observable
* Spaces (discrete)
    * States: [ height x width x channels ]
    * Observations: [ height x width x channels ]
        * agent POV
    * Action: int: 
* Objects do not stack
* Limited state features out of grid to agent direction and holding of object

# Implementation

## Dynamics

* In Minigrid: mostly in base abstract
* Mechanics:
    * Key & door
    * Button (door)
    * goal cell
    * dynamic obstacles
* Conceptually:
    1. Move agent / pickup / actuate
    1. Update objects in order (dynamic obstacles..?)
        * List through grid, call 'update' on each cell
* Requirements:
    * Move:
        * Blocks agent or not
    * Pickup
        * Can be picked up
    * Actuate:
        * `cell.actuate()`: state -> state?
    * Update
        * `update` function

## Cell objects

* `status`
* `color`
* `render()` -> `[type, property 1, property 2]`
* `can_be_picked_up` (or function?)
* `blocks_agent` (or function?)
* `actuate()`: state -> state (domain specific?)
* `update()`: state -> state (domain specific?)
* `transparent`

```python
class door():
    def actuate(state_grid, agent_direction: dir, hold_item: object):
        if hold_item.type == key and hold_item.color == self.shared_property:
            self.open()

class cell_object():
    @property
    def status() -> int:
        # e.g. door: return open/closed/locked
    
    def shared_property():
    # or
    def color():

class button():
    def actuate(state_grid, agent_direction: dir, hold_item: object):
        # loop over state_grid, find doors with self.shared_property -> open
```

## Environments

* `GymEnvironment(gym_api)` base base base class environment
    * API: functional
* `SimpleMiniGrid` <- all the specifications so far
