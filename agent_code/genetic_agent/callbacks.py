from collections import deque
import numpy as np

import agent_code.coin_hunter_agent.callbacks as coin_hunter_agent
import agent_code.wall_breaker.callbacks as wall_breaker_agent
import agent_code.rule_based_agent.callbacks as rule_based_agent
import agent_code.survival_agent.callbacks as survival_agent

HISTORY_LENGTH = 30

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
    self.damage_history = np.array([5, 5, 5, 5, 5, 5])

    # last HISTORY_LENGTH-step action history
    self.action_history = deque([], HISTORY_LENGTH)


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.action_history = deque([], HISTORY_LENGTH)


def check_if_actions_are_repeated(self, action):
    """
    Checks if two actions are repeated in the last HISTORY_LENGTH steps
    :param self:
    :param action: next action to be taken
    :return: random action if repeated, else action
    """

    if len(self.action_history) < HISTORY_LENGTH:
        return action

    # do not include the bomb in possible actions to return
    possible_actions_list = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]

    s = set(self.action_history)
    if len(s) == 2:
        count_1 = self.action_history.count(list(s)[0])
        count_2 = self.action_history.count(list(s)[1])
        if count_1 == count_2:
            print("REPEATED ACTION")

            # exclude list(s)[0] and list(s)[1] from possible actions
            possible_actions_list.remove(list(s)[0])
            possible_actions_list.remove(list(s)[1])

            return np.random.choice(possible_actions_list)

    return action


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

    name = game_state['agent_name']

    # pick agent weights
    weights = game_state['agent_weights']
    #print("AGENT WEIGHTS", weights)

    # get action and score from each agent set action = wait if the first argument is None
    if game_state is None:
        print("GAME STATE IS NONE", game_state)
    if self is None:
        print("SELF IS NONE", self)

    coin_hunter_params = {
        "reward_fun_weight": 0.05,  # the weight of the reward function
        "n_iters": 12,  # the number of iterations of the stochastic local beam search
        "max_hunters": 10,  # the maximum number of states of each iteration
    }
    wall_breaker_action_scores = wall_breaker_agent.behave(self, game_state)
    coin_hunter_action_scores = coin_hunter_agent.behave(game_state, coin_hunter_params)
    survival_agent_action_scores = survival_agent.behave(self, game_state)

    # if all scores are zero, return wait
    #if all(value == 0 for value in wall_breaker_action_scores.values()) \
    #        and all(value == 0 for value in coin_hunter_action_scores.values()) \
    #        and all(value == 0 for value in survival_agent_action_scores.values()):
    #    self.action_history.append("WAIT")
    #    action = check_if_actions_are_repeated(self, "WAIT")
    #    return action


    action_scores = {
        "wall_breaker": wall_breaker_action_scores,
        "survival": survival_agent_action_scores,
        "coin_hunter": coin_hunter_action_scores,
    }

    #if name == "genetic_agent_0":
    #    print(f"WEIGHTS: {weights}")
    #    print(f"ACTION SCORES: {action_scores}")

    #for key in action_scores:
    #    if name == "genetic_agent_0":
    #        print(f"genetic_agent_0 {key}: {action_scores[key]}")

    # for each agent, multiply each score on the dictionary by the weight
    for (agent, scores) in action_scores.items():
        for key in scores:
            scores[key] *= weights[agent]

    #if name == "genetic_agent_0":
    #    print(f"WEIGHTED ACTION SCORES: {action_scores}")

    # sum the scores of each action for each agent
    action_summed_scores = {
        "UP": 0,
        "RIGHT": 0,
        "DOWN": 0,
        "LEFT": 0,
        "WAIT": 0,
        "BOMB": 0,
    }
    for key in action_summed_scores:
        for (agent, scores) in action_scores.items():
            action_summed_scores[key] += scores[key]

    # get the action with the highest score
    max_score = 0
    max_action = None
    for key in action_summed_scores:
        if action_summed_scores[key] > max_score:
            max_score = action_summed_scores[key]
            max_action = key

    #if name == "genetic_agent_0":
        #print(f"SUMMED ACTION SCORES: {action_summed_scores}")
        #print(f"MAX ACTION: {max_action} with score {max_score}")

    #if name == "genetic_agent_0":
    #    print(f"action {max_action}")

    self.action_history.append(max_action)
    #action = check_if_actions_are_repeated(self, max_action)
    return max_action
