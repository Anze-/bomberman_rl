from collections import deque
from random import shuffle
from queue import PriorityQueue
import numpy as np
from items import Coin, Explosion, Bomb
import settings as s

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

def get_blast_coords(x,y, power, arena):
        blast_coords = [(x, y)]

        for i in range(1, power + 1):
            if arena[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, power + 1):
            if arena[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, power + 1):
            if arena[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, power + 1):
            if arena[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

        return blast_coords

def isValid(grid,node):
    if(node[0]<0 or node[1]<0 or node[0]>=grid.shape[0] or node[1]>=grid.shape[0]): return False
    if(grid[node[1],node[0]]==0):return False
    else: return True


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.debug(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~') 
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    self.logger.debug(f'Bombs array: {bombs}')

    
    
    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    #Survival agent
    #Detect enemies
    enemies=others
    arena_dim=arena.shape[0]
    state_value_matrix = np.matrix(np.ones((arena_dim,arena_dim)) * np.inf)

    #fill manhattan distance matrix from enemies (0 where enemies are located)
    for o in others:
        temp_value_matrix=np.zeros((arena_dim, arena_dim))
        for i in range(0, arena.shape[0]):
            for j in range(0, arena.shape[0]):
                temp_value_matrix[i,j]=temp_value_matrix[i,j]+ abs(i - o[1]) + abs(j - o[0])
        state_value_matrix=np.minimum(state_value_matrix,temp_value_matrix)
    
    self.logger.debug(f'Fill with manhattan from enemies distance matrix: \n{state_value_matrix}')
    #add 1 for each legal movement executable from cell i,j
    for i in range(0, arena.shape[0]-1):
            for j in range(0, arena.shape[0]-1):
                leg_move=0
                if(state_value_matrix[i,j]!=0):
                    if(state_value_matrix[i-1,j]!=0): leg_move+=1
                    if(state_value_matrix[i+1,j]!=0): leg_move+=1
                    if(state_value_matrix[i,j-1]!=0): leg_move+=1
                    if(state_value_matrix[i,j+1]!=0): leg_move+=1
                state_value_matrix[i,j]+=leg_move
    self.logger.debug(f'Add 1 to cell for each possible action: \n{state_value_matrix}')
    #put 0 where we have walls and crafts (non legal position)
    free_matrix=np.absolute(np.transpose(np.absolute(arena))-np.ones(arena.shape[0])) 
    state_value_matrix=np.multiply(state_value_matrix, free_matrix)

    
    
    #put 0 in bomb range if distance from agent to explosion cell is equal
    f_deadzone=game_state['dead_zones']
    for i in range(0, arena.shape[0]-1):
        for j in range(0, arena.shape[0]-1):
            dis=abs(j- x) + abs(i - y)
            deadzone=f_deadzone((j, i), dis)
            if deadzone:
                self.logger.debug(f'Agent location ({(x,y)}), cell location({j,i}) distance {dis}, deadzone {deadzone}, PUT 0') 
                state_value_matrix[i,j]=0

    #put 0 in bomb place if close to agent
    for (xb, yb), t in bombs:
        dis=abs(yb- y) + abs(xb - x)
        if dis<=4:
            self.logger.debug(f'Agent location ({(x,y)}), bomb location ({(xb,yb)}) distance {dis} <= {t}, PUT 0 ') 
            state_value_matrix[yb,xb]=0
    
    self.logger.debug(f'Put 0 in bomb place and in cells where path will encounter an explosion: \n{state_value_matrix}')   
    self.logger.debug(f'Agent location ({(x,y)})with value {state_value_matrix[y,x]}') 

    #put 0 in agent location if is in range of a bomb
    for (xb, yb), t in bombs:
        blast=get_blast_coords(xb, yb, 3, arena)
        self.logger.debug(f'Bomb  ({bombs}), range {blast} ') 
        if (x,y) in blast:
            self.logger.debug(f'Agent location ({(x,y)}) in bomb range, PUT 0 ') 
            state_value_matrix[y,x]=0

   