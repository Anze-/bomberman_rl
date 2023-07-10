from typing import Set

from enum import Enum
from collections import namedtuple


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