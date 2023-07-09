from collections import deque
import numpy as np
import sys
sys.path.append("agent_code")

import agent_code.coin_hunter_agent.callbacks as coin_hunter_agent
import agent_code.wall_breaker.callbacks as wall_breaker_agent
import agent_code.rule_based_agent.callbacks as rule_based_agent
import agent_code.survival_agent.callbacks as survival_agent

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


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to genetic agent logic')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    # pick agent weights
    weights = game_state['agent_weights']

    # get action and score from each agent set action = wait if the first argument is None
    if game_state is None:
        print("GAME STATE IS NONE", game_state)
    if self is None:
        print("SELF IS NONE", self)

    wall_breaker_action_scores = wall_breaker_agent.behave()
    coin_hunter_params = {
        "reward_fun_weight": 0.5,  # the weight of the reward function
        "n_iters": 10,  # the number of iterations of the stochastic local beam search
        "max_hunters": 5,  # the maximum number of states of each iteration
    }
    coin_hunter_action_scores = coin_hunter_agent.behave(game_state, coin_hunter_params)
    survival_agent_action_scores = survival_agent.behave()

    action_scores = [wall_breaker_action_scores, coin_hunter_action_scores, survival_agent_action_scores]

    # for each agent, multiply each score on the dictionary by the weight
    for i,  scores in enumerate(action_scores):
        for key in scores:
            action_scores[i][key] = action_scores[i][key] * weights[i]

    # get the action with the highest score
    max_score = 0
    max_action = None
    agent = None

    for i, scores in enumerate(action_scores):
        for key in scores:
            if scores[key] > max_score:
                max_score = scores[key]
                max_action = key
                if i%3 == 0:
                    agent = "wall_breaker"
                elif i%3 == 1:
                    agent = "coin_hunter"
                else:
                    agent = "survival"

    print(f"action {max_action} from agent {agent}")
    return max_action
