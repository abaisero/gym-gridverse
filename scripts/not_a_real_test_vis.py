from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import Colors, Door, Key, MovingObstacle
from gym_gridverse.state import Agent, Grid, State
from gym_gridverse.visualize import str_render_state

grid = Grid(2, 3)
agent = Agent((0, 0), Orientation.S, Key(Colors.BLUE))
state = State(grid, agent)

state.grid[1, 0] = MovingObstacle()
state.grid[0, 2] = Door(Door.Status.LOCKED, Colors.BLUE)

print(f"Init state:\n{str_render_state(state)}")
