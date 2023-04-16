from collections import deque
import numpy as np
import sys
sys.path.append("agent_code")

import agent_code.coin_collector_agent.callbacks as coin_collector_agent
import agent_code.wall_breaker.callbacks as wall_breaker_agent
import agent_code.rule_based_agent.callbacks as rule_based_agent

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
        print("game state", game_state)
    if self is None:
        print("self", self)

    wb_action, wb_score = wall_breaker_agent.act(self, game_state)
    coin_collector_action, coin_collector_score = coin_collector_agent.act(self, game_state)
    rule_based_action, rule_based_score = rule_based_agent.act(self, game_state)
    if wb_action is None:
        wb_action = "WAIT"
    if coin_collector_action is None:
        coin_collector_action = "WAIT"
    if rule_based_action is None:
        rule_based_action = "WAIT"


    # multiply elementwise the weights and the scores
    weighted_scores = np.multiply(weights, [wb_score, coin_collector_score, rule_based_score])
    # get the index of the max score
    output = np.argmax(weighted_scores)
    # return the action associated with the max score
    if output == 0:
        return wb_action
    if output == 1:
        return coin_collector_action
    if output == 2:
        return rule_based_action
