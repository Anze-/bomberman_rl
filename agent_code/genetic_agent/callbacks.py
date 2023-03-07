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


def compute_distance(objects, agent_x, agent_y, gui, name="", bomb=True):
    distance = 9999
    minx, miny = 9999, 9999

    for i, elem in enumerate(objects):
        if bomb:
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

    #if distance != 9999:
    #    if name == "genetic_agent_0":
    #        print(f"agent_x:{agent_x} agent_y:{agent_y} object_x:{minx} object_y:{miny} distance:{distance}")
    #    #gui.draw_line(agent_x, agent_y, minx, miny)

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
    name = game_state['agent_name']

    bombs_left = game_state["agent_bombs_left"]

    others = [xy for (n, s, b, xy) in game_state['others']]

    # Compute distance to bombs
    bomb_distance, coin_distance = 9999, 9999
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    arena = game_state['field']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    if bombs:
        bomb_distance = compute_distance(bombs, agent_x, agent_y, gui, name)

    coins_coords = game_state['coins_coords']
    if coins_coords:
        coin_distance = compute_distance(coins_coords, agent_x, agent_y, gui, name, bomb=False)

    # free nearest tiles

    # stay, right, left, down, up
    directions_list = [9999,9999,9999,9999,9999]

    directions = [(agent_x, agent_y), (agent_x + 1, agent_y), (agent_x - 1, agent_y), (agent_x, agent_y + 1), (agent_x, agent_y - 1)]
    valid_tiles, valid_actions = [], []
    for i, d in enumerate(directions):
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            directions_list[i] = 1


    if name == "genetic_agent_0":
        print(f"stay:{directions_list[0]} right:{directions_list[1]} left:{directions_list[2]} down:{directions_list[3]} up:{directions_list[4]}")

    net = game_state['agent_net']
    if net is not None:
        # !! attenzione modifica input ed output della rete in base a come viene settata la visione dell'agente qui sotto
        output = np.argmax(net.activate((bombs_left, bomb_distance, coin_distance,
                                         directions_list[1], directions_list[2], directions_list[3], directions_list[4])))

        if output == 0:
            return "RIGHT"
        if output == 1:
            return "LEFT"
        if output == 2:
            return "UP"
        if output == 3:
            return "DOWN"
        if output == 4:
            return "BOMB"
        if output == 5:
            return "WAIT"

    print("ERRORE, RITORNO AZIONE RANDOM")
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
