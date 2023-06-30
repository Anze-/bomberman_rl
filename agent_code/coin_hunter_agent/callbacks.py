from agent_code.coin_hunter_agent.movement import *
from agent_code.coin_hunter_agent.hunt import Hunter

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
    
    # perform a simple breadth-first search on the graph of currently viable+safe positions
    
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
def hunter_reward_fun(w : int):
    return lambda s: math.exp((-s**2) * w)


def act(self, game_state: dict) -> str:
    curr_pos = Coord(x=game_state['self'][3][0], y=game_state['self'][3][1])
    
    # perform a local beam search on the graph of positions with k=infty but limited number of steps,
    # exploiting the class Hunter as a representation of each state
    # and as the generator of every new successor state
    BEAM_SEARCH_ITERS = 5
    
    # initialize the pool of hunters with the initial state of the local search
    hunters = [
        Hunter(game_state, # the state of the game is the current
               game_state['coins'], # all coins currently in the field have not been collected yet
               curr_pos, # the position of the hunter is the current position of the agent
               hunter_reward_fun(0.02)
              )
    ]

    for _ in range(BEAM_SEARCH_ITERS):
        new_hunters = []
        
        # update the pool of hunter with all the successors of the current hunters in the pool
        for h in hunters:
            new_local_hunters = h.scatter()
            new_hunters.extend(new_local_hunters)
            
            # if a hunter has not generated successors, keep it in the pool
            if not new_local_hunters:
                new_hunters.append(h)
        
        hunters = new_hunters
    
    # find the hunter who scored the best utility
    best_hunter = hunters[0]
    for h in hunters:
        if h.utility > best_hunter.utility:
            best_hunter = h
    
    if best_hunter.utility == 0:
        return 'WAIT'
    return best_hunter.first_dir.name