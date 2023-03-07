from collections import deque
from random import shuffle
import math
import numpy as np

import settings as s


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def compute_distance(objects, agent_x, agent_y, gui):
    distance = 9999
    minx, miny = 9999, 9999
    for i, elem in enumerate(objects):
        res = type(elem) is tuple
        if res:
            xb = elem[0][0]
            yb = elem[0][1]
        else:
            xb = elem[0]
            yb = elem[1]

        d = math.sqrt((xb - agent_x) ** 2 + (yb - agent_y) ** 2)

        if d < distance:
            distance = d
            minx = xb
            miny = yb

    if distance != 9999:
        gui.draw_line(agent_x, agent_y, minx, miny)

    return distance


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to genetic agent logic')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    gui = game_state['gui']

    agent_x = game_state['agent_x']
    agent_y = game_state['agent_y']
    bombs_left = game_state["agent_bombs_left"]


    # Compute distance to bombs
    bomb_distance, coin_distance = 9999, 9999
    bombs = game_state['bombs']
    if bombs:
        bomb_distance = compute_distance(bombs, agent_x, agent_y, gui)

    coins = game_state['coins']
    if coins:
        coin_distance = compute_distance(coins, agent_x, agent_y, gui)

    net = game_state['agent_net']
    if net is not None:
        # !! attenzione modifica input ed output della rete in base a come viene settata la visione dell'agente qui sotto
        output = np.argmax(net.activate((agent_x, agent_y, bombs_left, bomb_distance, coin_distance)))
        print("output: ", output)

        if output == 0:
            print("RIGHT")
            return "RIGHT"
        if output == 1:
            print("LEFT")
            return "LEFT"
        if output == 2:
            print("UP")
            return "UP"
        if output == 3:
            print("DOWN")
            return "DOWN"
        if output == 4:
            print("BOMB")
            return "BOMB"
        if output == 5:
            print("WAIT")
            return "WAIT"

    print("ERRORE, RITORNO AZIONE RANDOM")
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
