from collections import deque
from random import shuffle
from queue import PriorityQueue
import numpy as np
from items import Coin, Explosion, Bomb
import settings as s
from typing import Dict, Optional, Tuple

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

def heuristic(a, b): # Manhattan distance
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def neighbors(node, grid):
    neigh=[]
    (x,y)=node
    for i in [-1, 1]:
        node1 = (x+i,y)
        node2 = (x,y+i)
        if isValid(grid,node1):
            neigh.append((node1))
        if isValid(grid,node2): 
            neigh.append((node2))
    return neigh

def reconstruct_path(came_from, start, goal):
    current= goal
    path= []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    #path.append(start) # optional
    path.reverse() # optional
    return path

def reacheable_nodes(grid, start, self):
    reacheable_grid=np.zeros((grid.shape[0],grid.shape[0]))
    frontier=[start]
    visited=[]
    while frontier:
        current= frontier.pop(0)
        visited.append(current)
        reacheable_grid[current[1],current[0]]=1
        neigh=neighbors(current, grid)
        for n in neigh:
            if n not in visited and n not in frontier:
                frontier.append(n)
    return reacheable_grid


def a_star_search(grid, start, goal, self):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from= {}
    cost_so_far= {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current= frontier.get()
        if current == goal:
            break
        neigh=neighbors(current, grid)
        for next in neigh:
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                h=heuristic(next, goal)
                priority = new_cost + h
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far

def print_path_cost(path,grid):
    costs=[]
    for node in path:
        (x,y)=node
        costs.append(grid[y,x])        
    return costs

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

    if not others: 
        state_value_matrix = state_value_matrix = np.matrix(np.ones((arena_dim,arena_dim)) * 20)

        #put 0 where we have walls and crafts (non legal position)
        free_matrix=np.absolute(np.transpose(np.absolute(arena))-np.ones(arena.shape[0])) 
        state_value_matrix=np.multiply(state_value_matrix, free_matrix)
    else:
        #fill manhattan distance matrix from enemies (0 where enemies are located)
        for o in others:
            temp_value_matrix=np.zeros((arena_dim, arena_dim))
            for i in range(0, arena.shape[0]):
                for j in range(0, arena.shape[0]):
                    temp_value_matrix[i,j]=temp_value_matrix[i,j]+ abs(i - o[1]) + abs(j - o[0])
            state_value_matrix=np.minimum(state_value_matrix,temp_value_matrix)
        
        self.logger.debug(f'Fill with manhattan from enemies distance matrix: \n{state_value_matrix}')

        #put 0 where we have walls and crafts (non legal position)
        free_matrix=np.absolute(np.transpose(np.absolute(arena))-np.ones(arena.shape[0])) 
        state_value_matrix=np.multiply(state_value_matrix, free_matrix)

        #add 1 for each legal movement executable from cell i,j
        #TODO: apply this only if cell value >=1 (so not in bomb range) and pt it at the end
        
        for i in range(0, arena.shape[0]-1):
                for j in range(0, arena.shape[0]-1):
                    leg_move=0
                    if(state_value_matrix[i,j]!=0):
                        if(state_value_matrix[i-1,j]!=0): leg_move+=1
                        if(state_value_matrix[i+1,j]!=0): leg_move+=1
                        if(state_value_matrix[i,j-1]!=0): leg_move+=1
                        if(state_value_matrix[i,j+1]!=0): leg_move+=1
                    state_value_matrix[i,j]+=leg_move    

    #put 0.3*t_until_expl and 0 in bomb range and bomb place if bomb closer then 8 cells to agent
    for (xb, yb), t in bombs:
        dis=abs(xb- x) + abs(yb - y)
        if dis<=8:
            blast=get_blast_coords(xb, yb, 3, arena)
            self.logger.debug(f'Bomb  ({bombs}), range {blast} ') 
            for (i, j) in blast:
                if state_value_matrix[j, i]!=0: 
                    state_value_matrix[j, i]=0.3*t
            state_value_matrix[yb, xb]=0

    #put 0 in bomb range if distance from agent to explosion cell is equal
    f_deadzone=game_state['dead_zones']
    for i in range(0, arena.shape[0]-1):
        for j in range(0, arena.shape[0]-1):
            dis=abs(j- x) + abs(i - y)
            deadzone=f_deadzone((j, i), dis)
            if deadzone:
                state_value_matrix[i,j]=0

    # preserve values of only reachable cells
    reacheable=reacheable_nodes(state_value_matrix, (x,y), self)

    state_value_matrix=np.multiply(state_value_matrix, reacheable)
    
    # Choose direction as min path to safer cell with a star
    
    goals = np.where(state_value_matrix == np.amax(state_value_matrix))
    listOfGoals = list(zip(goals[0], goals[1]))
    dis=50
    #Take the goal nearest to the agent
    for goal in listOfGoals:
        tempdis=heuristic((goal[1],goal[0]), (x,y))
        if (dis>tempdis):
            dis=tempdis
            true_goal=goal
    (x_goal,y_goal)=true_goal
    true_goal=(y_goal,x_goal)
    self.logger.debug(f'GOAL ({true_goal})') 
    came_from, cost_so_far=a_star_search(state_value_matrix, start=(x,y), goal=(true_goal), self=self)

    path=reconstruct_path(came_from, start=(x,y), goal=true_goal)
    
    if not path: #choose greedy action

        max_cell=(y,x)
        max=state_value_matrix[y,x]
        best_action='WAIT'
        self.logger.debug(f'Start from Agent cell ({(y,x)}) with value {state_value_matrix[y,x]}')
        if state_value_matrix[y+1,x]>max:
            max=state_value_matrix[y+1,x]
            max_cell=(y+1,x)
            best_action='DOWN' 
        if state_value_matrix[y-1,x]>max:
            max=state_value_matrix[y-1,x]
            max_cell=(y-1,x)
            best_action='UP'
        if state_value_matrix[y,x+1]>max:
            max=state_value_matrix[y,x+1]
            max_cell=(y,x+1)
            best_action='RIGHT'
        if state_value_matrix[y,x-1]>max:
            max=state_value_matrix[y,x-1]
            max_cell=(y,x-1)
            best_action='LEFT'
        self.logger.debug(f'Best_action cause no path: {best_action}') 
    else:
        if(path[0]==(x+1,y)):best_action= 'RIGHT'
        if(path[0]==(x-1,y)):best_action= 'LEFT'
        if(path[0]==(x,y+1)):best_action= 'DOWN'
        if(path[0]==(x,y-1)):best_action= 'UP'
        self.logger.debug(f'Best_action: {best_action}') 
    self.logger.debug(f'Final State Value Function with value \n{state_value_matrix}')
    return best_action

def behave(self, game_state: dict) -> Dict[str, float]:
    
    # initialize output scores
    action_scores = {
        'UP':    0.0,
        'DOWN':  0.0,
        'LEFT':  0.0,
        'RIGHT': 0.0,
        'BOMB':  0.0,
        'WAIT':  0.0,
    }
    
    """
    Called by the Superagent to get a set of action-score pairs of possible actions.
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

    #Return score of each move equal to 0 if no enemies alive
    if not others: 
        state_value_matrix = state_value_matrix = np.matrix(np.ones((arena_dim,arena_dim)) * 20)

        #put 0 where we have walls and crafts (non legal position)
        free_matrix=np.absolute(np.transpose(np.absolute(arena))-np.ones(arena.shape[0])) 
        state_value_matrix=np.multiply(state_value_matrix, free_matrix)
    else:
        #fill manhattan distance matrix from enemies (0 where enemies are located)
        for o in others:
            temp_value_matrix=np.zeros((arena_dim, arena_dim))
            for i in range(0, arena.shape[0]):
                for j in range(0, arena.shape[0]):
                    temp_value_matrix[i,j]=temp_value_matrix[i,j]+ abs(i - o[1]) + abs(j - o[0])
            state_value_matrix=np.minimum(state_value_matrix,temp_value_matrix)
        
        self.logger.debug(f'Fill with manhattan from enemies distance matrix: \n{state_value_matrix}')

        #put 0 where we have walls and crafts (non legal position)
        free_matrix=np.absolute(np.transpose(np.absolute(arena))-np.ones(arena.shape[0])) 
        state_value_matrix=np.multiply(state_value_matrix, free_matrix)

        #add 1 for each legal movement executable from cell i,j
        #TODO: apply this only if cell value >=1 (so not in bomb range) and pt it at the end
        
        for i in range(0, arena.shape[0]-1):
                for j in range(0, arena.shape[0]-1):
                    leg_move=0
                    if(state_value_matrix[i,j]!=0):
                        if(state_value_matrix[i-1,j]!=0): leg_move+=1
                        if(state_value_matrix[i+1,j]!=0): leg_move+=1
                        if(state_value_matrix[i,j-1]!=0): leg_move+=1
                        if(state_value_matrix[i,j+1]!=0): leg_move+=1
                    state_value_matrix[i,j]+=leg_move    

    #put 0.3*t_until_expl and 0 in bomb range and bomb place if bomb closer then 8 cells to agent
    for (xb, yb), t in bombs:
        dis=abs(xb- x) + abs(yb - y)
        if dis<=8:
            blast=get_blast_coords(xb, yb, 3, arena)
            self.logger.debug(f'Bomb  ({bombs}), range {blast} ') 
            for (i, j) in blast:
                if state_value_matrix[j, i]!=0: 
                    state_value_matrix[j, i]=0.3*t
            state_value_matrix[yb, xb]=0

    #put 0 in bomb range if distance from agent to explosion cell is equal
    f_deadzone=game_state['dead_zones']
    for i in range(0, arena.shape[0]-1):
        for j in range(0, arena.shape[0]-1):
            dis=abs(j- x) + abs(i - y)
            deadzone=f_deadzone((j, i), dis)
            if deadzone:
                state_value_matrix[i,j]=0

    # preserve values of only reachable cells
    reacheable=reacheable_nodes(state_value_matrix, (x,y), self)

    state_value_matrix=np.multiply(state_value_matrix, reacheable)
    # return best_action
    
    # Choose direction as min path to safer cell with a star
    
    goals = np.where(state_value_matrix == np.amax(state_value_matrix))
    listOfGoals = list(zip(goals[0], goals[1]))
    dis=50
    #Take the goal nearest to the agent
    for goal in listOfGoals:
        tempdis=heuristic((goal[1],goal[0]), (x,y))
        if (dis>tempdis):
            dis=tempdis
            true_goal=goal
    (x_goal,y_goal)=true_goal
    true_goal=(y_goal,x_goal)
    self.logger.debug(f'GOAL ({true_goal})') 
    came_from, cost_so_far=a_star_search(state_value_matrix, start=(x,y), goal=(true_goal), self=self)

    path=reconstruct_path(came_from, start=(x,y), goal=true_goal)
    
    if not path: #choose greedy action
        max_cell=(y,x)
        max=state_value_matrix[y,x]
        best_action='WAIT'
        self.logger.debug(f'Start from Agent cell ({(y,x)}) with value {state_value_matrix[y,x]}')
        if state_value_matrix[y+1,x]>max:
            max=state_value_matrix[y+1,x]
            max_cell=(y+1,x)
            best_action='DOWN' 
        if state_value_matrix[y-1,x]>max:
            max=state_value_matrix[y-1,x]
            max_cell=(y-1,x)
            best_action='UP'
        if state_value_matrix[y,x+1]>max:
            max=state_value_matrix[y,x+1]
            max_cell=(y,x+1)
            best_action='RIGHT'
        if state_value_matrix[y,x-1]>max:
            max=state_value_matrix[y,x-1]
            max_cell=(y,x-1)
            best_action='LEFT'
    else:
        if(path[0]==(x+1,y)):best_action= 'RIGHT'
        if(path[0]==(x-1,y)):best_action= 'LEFT'
        if(path[0]==(x,y+1)):best_action= 'DOWN'
        if(path[0]==(x,y-1)):best_action= 'UP'
        if(path[0]==(x,y)):best_action= 'WAIT'
    
    #Update the acton scores
    #Score will be 1/curr_value*increment_value, so the lower the value of current (cell security) the higher the score (need to move to another cell)
    curr_value=state_value_matrix[y,x]
    right_value=state_value_matrix[y,x+1] 
    left_value=state_value_matrix[y,x-1] 
    up_value=state_value_matrix[y-1,x] 
    down_value=state_value_matrix[y+1,x]   

    #First version with bias towards best action
    if best_action=='DOWN': action_scores['DOWN']=1/(curr_value+1)*((down_value-curr_value)+1)
    else: action_scores['DOWN']=1/(curr_value+1)*(down_value-curr_value)
    if best_action=='UP': action_scores['UP']=1/(curr_value+1)*((up_value-curr_value)+1)
    else: action_scores['UP']=1/(curr_value+1)*(up_value-curr_value)
    if best_action=='RIGHT': action_scores['RIGHT']=1/(curr_value+1)*((right_value-curr_value)+1)
    else: action_scores['RIGHT']=1/(curr_value+1)*(right_value-curr_value)
    if best_action=='LEFT': action_scores['LEFT']=1/(curr_value+1)*((left_value-curr_value)+1)
    else: action_scores['LEFT']=1/(curr_value+1)*(left_value-curr_value)
    if best_action=='WAIT': action_scores['WAIT']=1/(curr_value+1)*((curr_value-curr_value)+1)
    else: action_scores['WAIT']=1/(curr_value+1)*(curr_value-curr_value)

    #Zero score negative score actions
    if action_scores['DOWN']<0: action_scores['DOWN']=0
    if action_scores['UP']<0: action_scores['UP']=0
    if action_scores['RIGHT']<0: action_scores['RIGHT']=0
    if action_scores['LEFT']<0: action_scores['LEFT']=0
    if action_scores['WAIT']<0: action_scores['WAIT']=0
    
    return action_scores
    