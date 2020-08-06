# Gym Gridverse

Collection and library for grid-like domains.

# To do

* Array representation state and grid
* unittest (somehow) vis/manual control scripts
* More checks on whether generated elements are in space
* Increase supported (Minigrid) domains
* Add GUI Visualization
* Adapter for gym-usage
    * Have a bot walk around using gym interface
* Method to create more compact representation
* One-hot encoding of the state, action and observation (wrapper)
* Refactoring
    * Rename `StateDynamics` -> `TransitionFunction`
    * Move key/repeating functionality to grid
        * 'Object/pos in front of agent'
    * geometry module
* Re-produce minigrid behavior
    * Discounted reward
    * Minigrid state dynamics

## Environments

Taken from the [gym-minigrid](https://github.com/maximecb/gym-minigrid)

* [x] Empty (size + random flag)
* [x] 4 room (random agent/goal position flag)
* [ ] Door & key
* [ ] Multi-room
    * non-trivial to generate
* [ ] Red/Blue Door
    * Requires reward/termination function for opening (all?) doors
* [ ] Memory
    * Requires reward/termination function for 'going to matching object'
* [ ] Key corridor
    * Sounds like a lot of effort to create (hard coding?)
* [ ] Unlock
    * + pickup -> requires reward/termination function for picking up something
    * + pickup & blocked -> also requires 'box' object
* [ ] Obstructed maze
    * Sounds like a lot of hard coding
* [ ] Shift
    * Requires lava (and appropriate reward/termination functions)
* [ ] Lava gap + extended lava crossing
    * Requires lava (and appropriate reward/termination functions)
    * Plenty of hard coding (or non-trivial generation)
* [x] Dynamic obstacles

# Representation

The state is represented by two pieced, a grid and agent information. These
have an analogue representation as observation. An action is a glorified
integer.

## Grid

* A width by height matrix of objects, where an object is represented by:
    * A type (integer)
    * A status (integer, meaning dependent on type)
    * A color

## Agent

* Position (x,y)
* Orientation (N/E/S/W represented by int)
* Holding object (with same properties as those in the grid)

## Objects

As mentioned above, an object is described by its type, status and color. We
provide the following objects out of the box:

* `Floor`: the most basic (empty) grid cell
* `Wall`: a blocking (non-transparent) cell
* `Goal`: Cell designated to represent rewarding and terminating goals
* `Door`: Either open, closed or locked. Unlocked by `Key` if same color
* `Key`: An object that can be picked up and unlocked `Door` with same color
* `MovingObstacle`: An obstacle that moves stochastically over `Floor`

## Observation

An array/integer representation of a slice of the grid and the agent state.
Since some objects are not transparent, a object `Hidden` represents cells are
not observable. The observation is always from the point of view of the agent.
We provide the following observation functions:

* Mimic like gym-minigrid
* A more intuitive non-transparent blocking observation where a direct line of
  sight is necessary to observe a cell
* Stochastic version of the observation function

## Actions:

1. move forward
1. move backward
1. move left
1. move right
1. turn left
1. turn right
1. actuate
1. pick and drop
