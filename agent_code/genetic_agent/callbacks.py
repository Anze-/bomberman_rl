from collections import deque
import math
import numpy as np
import sys
sys.path.append("agent_code")

import agent_code.coin_collector_agent.callbacks as coin_collector_agent
import agent_code.survival_agent.callbacks as survival_agent
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


def from_action_to_id(action, agent):
    if action == "RIGHT":
        return 0
    if action == "LEFT":
        return 1
    if action == "UP":
        return 1
    if action == "DOWN":
        return 3
    if action == "BOMB":
        return 4
    if action == "WAIT":
        return 5
    else:
        print(f"ERROR: action {action} not found from agent {agent}")
        return 6
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

    survival_action = survival_agent.act(self, game_state)
    coin_collector_action = coin_collector_agent.act(self, game_state)
    rule_based_action = rule_based_agent.act(self, game_state)

    survival_action_number = from_action_to_id(survival_action)
    coin_collector_action_number = from_action_to_id(coin_collector_action)
    rule_based_action_number = from_action_to_id(rule_based_action)

    net = game_state['agent_net']
    if net is not None:
        output = np.argmax(net.activate((survival_action_number, coin_collector_action_number, rule_based_action_number)))

        if output == 0:
            return survival_action
        if output == 1:
            return coin_collector_action
        if output == 2:
            return rule_based_action

    print("ERRORE, RITORNO AZIONE RANDOM")
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
