from typing import Optional, Set

from enum import Enum
from collections import deque, namedtuple


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Coord = namedtuple('Coord', ['x', 'y'])

# note: the origin coord (0, 0) is at the top-left corner of the arena
class Dir(Enum):
    UP    = Coord( 0, -1)
    DOWN  = Coord( 0,  1)
    LEFT  = Coord(-1,  0)
    RIGHT = Coord( 1,  0)

# game defined values representing what is present in a field tile
FIELD_WALL = -1
FIELD_FREE = 0
FIELD_CRATE = 1


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def apply_dir(pos: Coord, d: Dir) -> Coord:
    return Coord(pos.x + d.value.x, pos.y + d.value.y)


# returns true iff the given position is currently free from bombs and rivals, thus viable
def is_pos_free(pos: Coord, game_state: dict) -> bool:
    enemies = [pos for (_, _, _, pos) in game_state['others']]
    bombs =   [pos for (pos, _) in game_state['bombs']]
    
    return (game_state['field'][pos] == FIELD_FREE) and (pos not in bombs) and (pos not in enemies)


def is_pos_safe(pos: Coord, game_state: dict, t: int):
    return not game_state['dead_zones'](pos, t)


# returns the set of safe directions that allow for a movement from the given position
def get_valid_dirs(pos: Coord, game_state: dict, t: int) -> Set[Dir]:
    valid_dir = set()
    
    for d in Dir:
        new_pos = apply_dir(pos, d)
        if is_pos_free(new_pos, game_state) and is_pos_safe(new_pos, game_state, t):
            valid_dir.add(d)
            
    return valid_dir

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


def act(self, game_state: dict) -> str:
    curr_pos = Coord(x=game_state['self'][3][0], y=game_state['self'][3][1])
    
    d = find_dir_to_closest(game_state, curr_pos)
    if not d:
        return 'WAIT'
    return d.name