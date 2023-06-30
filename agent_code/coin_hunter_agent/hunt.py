from agent_code.coin_hunter_agent.movement import *

from typing import Callable, MutableSet, Optional


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# A hunter represents the state of a local search performed in the graph of currently viable+safe positions
#   - it cannot loop back to previously visited positions, unless a new coin have been collected
#   - it keeps track of the utility of its traveled path, defined as the sum of values
#     attributed to each collected coin by a given reward function, which operates wrt
#     the coin's distance along the path
class Hunter:
    
    def __init__(self,
                 game_state: dict,
                 available_coins: MutableSet[Coord],
                 pos: Coord, # the current (last) position along the traveled path
                 reward_fun: Callable[[int], float], # maps: coin distance along path -> coin value
                 visited: MutableSet[Coord] = set(),
                 steps: int = 0,
                 utility: float = 0.0,
                 first_dir: Optional[Dir] = None # the first direction taken along the traveled path
                ):
        
        self.game_state = game_state
        self.available_coins = available_coins.copy()
        self.visited = visited.copy()
        self.pos = pos
        self.reward_fun = reward_fun
        self.steps = steps
        self.utility = utility
        self.first_dir = first_dir
        
        if self.pos in self.available_coins:
            # the hunter found a coin in its current position
            self.available_coins.remove(self.pos)
            self.visited.clear()
            self.utility += self.reward_fun(self.steps)
        
        self.visited.add(self.pos)
    
    
    # scatters a new hunter to every possibile next position
    # i.e. generates and returns each eligible next state of the local search
    def scatter(self) -> list['Hunter']:
        
        # instantiate a new hunter for every next valid+safe+unvisted position
        new_hunters = []
        for d in get_valid_dirs(self.pos, self.game_state, self.steps + 1):
            
            next_pos = apply_dir(self.pos, d)
            if next_pos in self.visited:
                continue
            
            # the "first_dir" of the new hunter is preserved, except when "self" is the root hunter,
            # in which case it is assigned to the currently considered direction
            new_first_dir = self.first_dir
            if not new_first_dir:
                new_first_dir= d
            
            h = Hunter(self.game_state,
                       self.available_coins,
                       next_pos,
                       self.reward_fun,
                       self.visited,
                       self.steps + 1,
                       self.utility,
                       new_first_dir
                      )
            new_hunters.append(h)
        
        return new_hunters