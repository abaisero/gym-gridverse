from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Colors, Door, Key, MovingObstacle
from gym_gridverse.state import Agent, Grid, State
from gym_gridverse.visualize import str_render_state

grid = Grid(2, 3)
agent = Agent(Position(0, 0), Orientation.S, Key(Colors.BLUE))
state = State(grid, agent)

state.grid[Position(1, 0)] = MovingObstacle()
state.grid[Position(0, 2)] = Door(Door.Status.LOCKED, Colors.BLUE)

print(f"Init state:\n{str_render_state(state)}")
