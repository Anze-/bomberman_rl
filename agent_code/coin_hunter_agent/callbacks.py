from agent_code.coin_hunter_agent.movement import *
from agent_code.coin_hunter_agent.hunt import Hunt
import numpy as np
from typing import Dict, Optional, Tuple

import math
from collections import deque


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
    self.damage_history = np.array([5, 5, 5, 5, 5, 5])


# given a starting position, returns the direction and the distance towards the closest
# safely reachable coin, or None when such coin do not exists (or when it is in the given position)
def find_dir_to_closest(game_state: dict, start_pos: Coord) -> Optional[Tuple[Dir, int]]:
    
    # no coins in the field, return immediately
    if not game_state['coins']:
        return None
    
    # perform a simple breadth-first search on the graph of currently viable+safe paths
    
    frontier = deque([start_pos])
    parent_of = {start_pos: start_pos} # represents the search tree + keeps track of already visited positions
    t = 1 # tracks how many steps have been taken (== depth of the breadth-first expanded search tree)
    found = False
    
    while frontier:
        curr_pos = frontier.popleft()
        
        if curr_pos in game_state['coins']: # we found a coin: it must be the closest!
            found = True
            break
        
        successors = []
        for d in get_valid_dirs(curr_pos, game_state, t):
            successors.append(apply_dir(curr_pos, d))
        
        t = t + 1
        for next_pos in successors:
            if next_pos in parent_of: # the successor was already visited: avoid loops!
                continue
            parent_of[next_pos] = curr_pos
            frontier.append(next_pos)
    
    # no coin is currently safely reachable
    if not found:
        return None
    
    # the coin was actually in the given position
    if start_pos == curr_pos:
        return (None, 0)
    
    # the coin have been found: backtrack the search tree until its first level
    while parent_of[curr_pos] != start_pos:
        curr_pos = parent_of[curr_pos]
    
    # return the direction that the BFS has taken as first step toward the coin
    return (Dir(Coord(curr_pos.x - start_pos.x, curr_pos.y - start_pos.y)), t)


# given a weight w>0, returns a reward function for the collection of coins, r(s):e^(-w*s^2),
# mapping coin distance along path -> coin value
def coin_reward_fun(w : int):
    return lambda s: math.exp((-s**2) * w)


# given the current state of the game and a set of parameters of the coin hunter behaviour,
# returns the set of scores [0.0, 1.0] of how good each possible next action would be
# according to the behaviour;
# coin hunter behaviour parameters:
# - reward_fun_weight: the positive weight w of the coin_reward_fun
# - n_iters: the number of iterations of the stochastic local beam search
# - max_hunters: the maximum number of states of each iteration
def behave(game_state: dict, params: dict) -> Dict[str, float]:
    # initialize output scores
    action_scores = {
        Dir.UP.name:    0.0,
        Dir.DOWN.name:  0.0,
        Dir.LEFT.name:  0.0,
        Dir.RIGHT.name: 0.0,
        'BOMB':         0.0,
        'WAIT':         0.0,
    }
    
    # no coin to collect: no particular action to perform
    if not game_state['coins']:
        return action_scores
    
    curr_pos = Coord(x=game_state['self'][3][0], y=game_state['self'][3][1])
    visited_pos = set([curr_pos])
    reward_fun = coin_reward_fun(params['reward_fun_weight'])
    successful_hunt = False
    
    # compute the best possible utility that may be found by a hunt
    # (== utility of the path of length n_iters rewarding a new coin at each step)
    max_utility = 0
    for step in range(params['n_iters']):
        max_utility += reward_fun(step)
    
    # from every possible+safe next position, perform a hunt for coins
    for d in get_valid_dirs(curr_pos, game_state, 1):
        starting_pos = apply_dir(curr_pos, d)
        hunt = Hunt(game_state, starting_pos, reward_fun, params['max_hunters'], visited_pos)
        hunt.run(params['n_iters'])
        # assign as a score to the current considered direction the utility of the best path found,
        # rescaled to the interval [0,1] 
        action_scores[d.name] = hunt.best_utility() / max_utility
        if action_scores[d.name] > 0:
            successful_hunt = True
    
    # the hunt found no coins: resort to BFS over the entire field
    # note: if not even BFS finds a path to the coins, it means they are currently unreachable
    if not successful_hunt:
        bfs_result = find_dir_to_closest(game_state, curr_pos)
        if bfs_result and bfs_result[0]:
            # BFS found a path towards a coin: assign the lowest possible score to its direction
            action_scores[bfs_result[0].name] = reward_fun(params['n_iters'] + 1) / max_utility
        
    return action_scores


def act(self, game_state: dict) -> str:
    action_scores = behave(game_state, {
        'reward_fun_weight': 0.05,
        'max_hunters': 12,
        'n_iters': 10
    })
    
    # simply return the action that the behaviour scored as the best
    best_dir = None
    best_score = 0
    for d in Dir:
        if action_scores[d.name] > best_score:
            best_score = action_scores[d.name]   
            best_dir = d
    
    # if all action are scored 0, stay still
    if not best_dir:
        return 'WAIT'
    return best_dir.name