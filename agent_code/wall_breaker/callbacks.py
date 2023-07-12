from collections import deque
from random import shuffle
import copy

import numpy as np


class Behaviour:
    def __init__(self):
        ...

    def behave(self):
        action = "WAIT"
        score = 0
        return action, score

    def act(self, *args, **kwargs):
        action, _ = self.behave(self, args)
        return action



import settings as s

# the callback functions used by this agent are 4:
#      look_for_targets : find closest target
#                 setup : initializes data structures before a set of games
#            reset_self : called once at the beginning of each round
#                   act : called at each step to perform next move


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest block.

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
    if len(targets) == 0:
        return None

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
    if logger:
        logger.debug(f"Suitable target found at {best}")
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug("Successfully entered setup code")
    self.current_round = 0
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.first_bomb = False
    self.bomb_history = deque([], 5)
    self.damage_history=np.array([5,5,5,5,5,5])


def reset_self(self):
    pass


def bomb_damage(bombxy, gamemap, safemap, r=3):
    #local beam search implementation
    #r is the bomb radius
    x, y = bombxy
    #print(x, y)
    col = gamemap[x, max(0, y - r):min(16, y + r + 2)].flatten()
    row = gamemap[max(0, x-r):max(16, x+r+2), y].flatten()

    #do not count after bricks
    col[3] = 747
    row[3] = 747
    csplits = np.split(col, np.where(col == -1)[0])
    rsplits = np.split(row, np.where(row == -1)[0])
    for el in csplits:
        if 747 in el:
            csplit = el

    for el in rsplits:
        if 747 in el:
            rsplit = el

    damage = np.sum(
        1 == np.concatenate([
            rsplit,
            csplit,
        ])
    )
    safemap[x, max(0,y-r):min(16,y+r+2)] = -9
    safemap[max(0,x-r):max(16,x+r+2), y] = -9
    safety = (safemap == 10).sum()
    #print(safemap)
    #print("safety", safety)
    #print([gamemap[x, y-r:y+r+2], gamemap[x-r:x+r+2, y]])
    #import pdb
    #pdb.set_trace()
    #avoid suicide bombing
    if safety == 0:
        damage = 0

    return damage, safety, safemap
def recursive_accessible_area(myxy, mymap, counter, threshold=16):
    if counter>threshold:
        return mymap
    counter += 1
    x, y = myxy
    mymap[x, y] = 10
    neighbours = copy.deepcopy(mymap)[x-1:x+2, y-1:y+2]
    #print(neighbours)
    neighbours[0, 0] = -1
    neighbours[0, 2] = -1
    neighbours[2, 0] = -1
    neighbours[2, 2] = -1

    steps = np.vstack(np.where(neighbours == 0)).T
    #print(mymap)
    for xy in steps:
        #print(myxy, xy)
        newxy = myxy + xy - 1
        #print(myxy, xy, newxy)
        x, y = newxy
        mymap[x, y] = 10
    for xy in steps:
        newxy = myxy + (xy - 1)
        recursive_accessible_area(newxy, mymap, counter, threshold)

    return mymap

def open_area(myxy, gamemap):
    mymap = copy.deepcopy(gamemap)
    mymap = recursive_accessible_area(myxy, mymap, 0, threshold=16)
    myarea = (mymap == 10).sum()

    return mymap, myarea


def get_score(myarea, damage, safety):
    #print("----")
    # fraction of the map accessible to the user
    strategic_control = myarea/256
    # fraction of the theoretical max damage
    power = damage/12
    # 0 when 0 cells to survive 1 when all cells survive
    norm_safety = safety/myarea
    score = (1-strategic_control)*((0.9*power)+(0.1*norm_safety))
    #score = (1-strategic_control)*power
    #print(strategic_control,power,norm_safety)
    #print((1-strategic_control),(0.9*power),(0.1*norm_safety))
    #print(score)
    #print("----")


    #the score of the move dicreases when the available area is larger
    #increases when the bomb can destroy more blocks
    #decreases when the move is dangerous
    return score


def act(self, game_state):
    # calculate the player available game area
    # calculate the damage of the current position
    # move towards the best position to place a bomb
    # if current position has good damage place a bomb
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """

    #
    # import pdb
    # pdb.set_trace()
    self.logger.info("Picking action according to rule set")
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state["field"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state["others"]]
    coins = game_state["coins"]
    myxy = game_state['self'][3]
    x, y = myxy

    # random walk
    available_moves = []
    if arena[x+1,y] == 0: available_moves.append("RIGHT")
    if arena[x-1, y] == 0: available_moves.append("LEFT")
    if arena[x, y-1] == 0: available_moves.append("UP")
    if arena[x, y+1] == 0: available_moves.append("DOWN")
    walk = np.random.choice(available_moves)


    # compute heuristics
    accmap, myarea = open_area(myxy, arena)
    safemap = copy.deepcopy(accmap)
    damage, safety, safemap = bomb_damage(myxy, arena, safemap, r=3)
    #print(safemap)
    self.damage_history = self.damage_history[1:]
    self.damage_history = np.append(self.damage_history,damage)
    #print(myarea,self.damage_history, get_score(myarea, damage, safety))
    #if damage == max(self.damage_history) and damage>0:
    #    self.damage_history = self.damage_history*0+5
    #    self.damage_history = [999,999,999,999,999]
    #    print(damage,"=> BOMB",)
    #    return "BOMB"

    # if the best damage in the last n turns suggest to place a bomb

    # too slow!
    scd = brick_walk(self, game_state, myarea, arena, safemap, accmap, myxy)
    # print(scd)
    return list(scd.keys())[np.argmax(list(scd.values()))]



def random_walk(arena):
    score_dict = {
            "WAIT": 0,
            "BOMB": 0,
            "UP": 0,
            "DOWN": 0,
            "RIGHT": 0,
            "LEFT": 0,
        }

    # random walk
    available_moves = []
    if arena[x+1, y] == 0: available_moves.append("RIGHT")
    if arena[x-1, y] == 0: available_moves.append("LEFT")
    if arena[x, y-1] == 0: available_moves.append("UP")
    if arena[x, y+1] == 0: available_moves.append("DOWN")
    score_dict[np.random.choice(available_moves)] = 0.05
    return score_dict


def best_bomb(accmap):
    heumap = np.zeros([17, 17])
    for P in np.ndenumerate(accmap):
        (x, y), val = P
        if val == 10:
            safemap = copy.deepcopy(accmap)
            damage, safety, safemap = bomb_damage([x, y], accmap, safemap, r=3)
            #print(x,y,damage)
            heumap[x, y] = damage

    best_bomb_xy = np.unravel_index(np.argmax(heumap), heumap.shape)
    #print(heumap)
    #print("best bomb xy: ",  best_bomb_xy)
    return best_bomb_xy, heumap


def dijkstra(accmap, myxy, bombxy):
    from dijkstra import Graph, DijkstraSPF
    # create the graph
    graph = Graph()
    for P in np.ndenumerate(accmap):
        (x, y), val = P
        if val == 10:
            try:
                if accmap[x+1, y] == 10:
                    #print(f"{x},{y}", " <-> ", f"{x+1},{y}")
                    graph.add_edge(f"{x},{y}", f"{x+1},{y}", 1)
                    graph.add_edge(f"{x+1},{y}", f"{x},{y}", 1)
                if accmap[x, y+1] == 10:
                    #print(f"{x},{y}", " <-> ", f"{x},{y+1}")
                    graph.add_edge(f"{x},{y}", f"{x},{y+1}", 1)
                    graph.add_edge(f"{x},{y+1}", f"{x},{y}", 1)
            except:
                #print("out of range")
                pass
    x, y = myxy
    bx, by = bombxy
    dijk = DijkstraSPF(graph, f"{x},{y}")
    #import pdb
    #pdb.set_trace()
    path = dijk.get_path(f"{bx},{by}")
    return path


def brick_walk(self, game_state, myarea, arena, safemap, accmap, myxy):
    score_dict = {
            "WAIT": 0,
            "BOMB": 0,
            "UP": 0,
            "DOWN": 0,
            "RIGHT": 0,
            "LEFT": 0,
        }

    # check we have something to break otherwise exit now
    if (accmap == 1).sum() < 1:
        return score_dict

    bombxy, heumap = best_bomb(accmap)

    #we are on the target => drop the bomb
    if bombxy==myxy:
        # print("=======")
        # print(myxy)
        # print(accmap)
        # print(heumap)
        #import pdb
        #pdb.set_trace()
        damage, safety, safemap = bomb_damage(myxy, arena, safemap, r=3)
        score_dict["BOMB"] = get_score(myarea, damage, safety)
        self.damage_history = [999, 999, 999, 999, 999, 999, 999]
        return score_dict

    best_damage = heumap.max()/12

    bestpath = dijkstra(accmap, myxy, bombxy)
    distance = len(bestpath)

    move_score = best_damage/distance
    #import pdb
    #pdb.set_trace()
    if len(bestpath)<2:
        nextxy = myxy
    else:
        nextxy = list(map(int, bestpath[1].split(",")))

        #check if the suggestion is immediately deadly
        if game_state["dead_zones"](tuple(nextxy), game_state["step"] + 1):
            return score_dict
    move = np.array(nextxy) - np.array(myxy)
    if (move == [-1,  0]).all(): score_dict["LEFT"] = move_score
    if (move == [ 1,  0]).all(): score_dict["RIGHT"] = move_score
    if (move == [ 0, -1]).all(): score_dict["UP"] = move_score
    if (move == [ 0,  1]).all(): score_dict["DOWN"] = move_score

    return score_dict


def behave(self, game_state):
    # calculate the player available game area
    # calculate the damage of the current position
    # if current position has good damage place a bomb
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """

    #
    # import pdb
    # pdb.set_trace()
    self.logger.info("Picking action according to rule set")
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state["field"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state["others"]]
    coins = game_state["coins"]
    myxy = game_state['self'][3]
    x, y = myxy

    # random walk
    available_moves = []
    if arena[x+1,y] == 0: available_moves.append("RIGHT")
    if arena[x-1, y] == 0: available_moves.append("LEFT")
    if arena[x, y-1] == 0: available_moves.append("UP")
    if arena[x, y+1] == 0: available_moves.append("DOWN")
    walk = np.random.choice(available_moves)


    # compute heuristics
    accmap, myarea = open_area(myxy, arena)
    safemap = copy.deepcopy(accmap)
    damage, safety, safemap = bomb_damage(myxy, arena, safemap, r=3)
    self.damage_history = self.damage_history[1:]
    self.damage_history = np.append(self.damage_history,damage)
    #print(myarea, self.damage_history)

    # if the best damage in the last n turns suggest to place a bomb
    #if damage == max(self.damage_history) and damage > 0:
        # self.damage_history = self.damage_history*0+5
        # self.damage_history = [999,999,999,999,999,999,999]
        # return {
        #     "WAIT": 0,
        #     "BOMB": get_score(myarea, damage, safety),
        #     "UP": 0,
        #     "DOWN": 0,
        #     "RIGHT": 0,
        #     "LEFT": 0,
        # }

    # do not suggest to go back on the bomb just placed

    if 999 == max(self.damage_history):
        return {
            "WAIT": 0,
            "BOMB": 0,
            "UP": 0,
            "DOWN": 0,
            "RIGHT": 0,
            "LEFT": 0,
        }
    else:
        # this is the normal case, tell me how to go to the closest bomb
        return brick_walk(self, game_state, myarea, arena, safemap, accmap, myxy) # random_walk(arena)

