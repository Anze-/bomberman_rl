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
    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# performs a local beam search on the graph of currently viable+safe paths
# with k=infty but limited number of steps,
# exploiting the class Hunter as a representation of each state
# and as the generator of every new successor state
class Hunt:
    
    def __init__(self,
                 game_state: dict,
                 initial_pos: Coord,
                 reward_fun: Callable[[int], float]
                ):
    
        self.game_state = game_state
        self.initial_pos = initial_pos
        self.reward_fun = reward_fun
    
    
    def run(self, n_iters: int) -> list[Hunter]:
        
        hunters = [
            Hunter(self.game_state, # the state of the game is the current
                   self.game_state['coins'], # all coins currently in the field have not been collected yet
                   self.initial_pos, # the position of the hunter is the current position of the agent
                   self.reward_fun
                  )
        ]

        for _ in range(n_iters):
            new_hunters = []
            
            # update the pool of hunter with all the successors of the current hunters in the pool
            for h in hunters:
                new_local_hunters = h.scatter()
                new_hunters.extend(new_local_hunters)
                
                # if a hunter has not generated successors, keep it in the pool
                if not new_local_hunters:
                    new_hunters.append(h)
            
            hunters = new_hunters
            
        return hunters