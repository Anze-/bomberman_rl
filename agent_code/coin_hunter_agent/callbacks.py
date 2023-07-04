from agent_code.coin_hunter_agent.movement import *
from agent_code.coin_hunter_agent.hunt import Hunt, Hunter

from typing import Optional

import math
from collections import deque


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def setup(self):
    pass


# given a starting position, returns the direction towards the closest safely reachable coin
# or None when such coin do not exists (or when it is in the given position)
def find_dir_to_closest(game_state: dict, start_pos: Coord) -> Optional[Dir]:
    if not game_state['coins']:
        return None
    
    # perform a simple breadth-first search on the graph of currently viable+safe paths
    
    frontier = deque([start_pos])
    parent_of = {start_pos: start_pos} # represents the search tree + keeps track of already visited positions
    t = 1 # tracks how many steps have been taken (== depth of the breadth-first expanded search tree)
    
    while frontier:
        curr_pos = frontier.popleft()
        
        if curr_pos in game_state['coins']: # we found a coin: it must be the closest!
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
    
    # the coin was actually in the given position
    if start_pos == curr_pos:
        return None
    
    # the coin have been found: backtrack the search tree until its first level
    while parent_of[curr_pos] != start_pos:
        curr_pos = parent_of[curr_pos]
    
    # return the direction that the BFS has taken as first step toward the coin
    return Dir(Coord(curr_pos.x - start_pos.x, curr_pos.y - start_pos.y))


# given a weight w>0, returns a reward function for the collection of coins,
# mapping coin distance along path -> coin value
def coin_reward_fun(w : int):
    return lambda s: math.exp((-s**2) * w)


def act(self, game_state: dict) -> str:
    # no coin to collect: stay still
    if not game_state['coins']:
        return 'WAIT'
    
    # initiate a hunt for coins starting from the current position of the agent
    BEAM_SEARCH_ITERS = 5
    curr_pos = Coord(x=game_state['self'][3][0], y=game_state['self'][3][1])
    hunt = Hunt(game_state, curr_pos, coin_reward_fun(0.02))
    hunter_pool = hunt.run(BEAM_SEARCH_ITERS)
    
    # find the hunter who scored the best utility, and get the first direction of its path
    best_hunter = hunter_pool[0]
    for h in hunter_pool:
        if h.utility > best_hunter.utility:
            best_hunter = h
    best_dir = best_hunter.first_dir
    
    # if the best hunter reached some coins, resort to BFS over the entire field
    if best_hunter.utility == 0:
        best_dir = find_dir_to_closest(game_state, curr_pos)
    
    # if not even BFS found a path to the coins, it means they are currently unreachable
    if not best_dir:
        return 'WAIT'
    return best_dir.name