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


def compute_distance(agent_x, agent_y, point_x, point_y):
    dist = math.sqrt((point_x - agent_x) ** 2 + (point_y - agent_y) ** 2)
    return dist


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

    agent_x = game_state['agent_x']
    agent_y = game_state['agent_y']
    bombs_left = game_state["agent_bombs_left"]
    coins = game_state['coins']

    # Compute distance to bombs
    bomb_distance = 9999
    bombs = game_state['bombs']
    if bombs != []:
        for i, elem in enumerate(bombs):

            xb = elem[0][0]
            yb = elem[0][1]
            d = compute_distance(agent_x, agent_y, xb, yb)
            if d < bomb_distance:
                bomb_distance = d


    net = game_state['agent_net']
    if net is not None:
        # !! attenzione modifica input ed output della rete in base a come viene settata la visione dell'agente qui sotto
        output = np.argmax(net.activate((agent_x, agent_y, bombs_left, bomb_distance)))
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

    """state = {
        'round': self.round,
        'step': self.step,
        'field': np.array(self.arena),
        'self': agent.get_state(),
        'others': [other.get_state() for other in self.active_agents if other is not agent],
        'bombs': [bomb.get_state() for bomb in self.bombs],
        'coins': [coin.get_state() for coin in self.coins if coin.collectable],
        'user_input': self.user_input,
    }
    
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]

    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))"""
