import copy
import random as rd
import math
import numpy as np
from collections import defaultdict


class RLKey:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __hash__(self):
        return hash((tuple(frozenset(x) for x in self.k1),
                tuple(frozenset(x) for x in self.k2)))

    def __eq__(self, other):
        return (self.k1, self.k2) == (other.k1, other.k2)

    def __str__(self):
        return str(self.k1)+'|'+str(self.k2)



class WolfGoatCabbageQLearning:

    def __init__(self, start_state, goal_state, gamma=0.8, max_episodes=50000,
                 epsilon_greedy=True, min_epsilon=0.1, max_epsilon=1.0):
        self.start_state = tuple(set(x) for x in start_state)
        self.goal_state = tuple(set(x) for x in goal_state)

        self.gamma = gamma
        self.max_episodes = max_episodes

        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = 0.02
        self.epsilon = self.max_epsilon
        self.epsilon_greedy = epsilon_greedy

        self.actions = [
            'MOVE_CABBAGE_FROM_LEFT_TO_BOAT',
            'MOVE_CABBAGE_FROM_BOAT_TO_LEFT',
            'MOVE_CABBAGE_FROM_RIGHT_TO_BOAT',
            'MOVE_CABBAGE_FROM_BOAT_TO_RIGHT',
            'MOVE_GOAT_FROM_LEFT_TO_BOAT',
            'MOVE_GOAT_FROM_BOAT_TO_LEFT',
            'MOVE_GOAT_FROM_RIGHT_TO_BOAT',
            'MOVE_GOAT_FROM_BOAT_TO_RIGHT',
            'MOVE_WOLF_FROM_LEFT_TO_BOAT',
            'MOVE_WOLF_FROM_BOAT_TO_LEFT',
            'MOVE_WOLF_FROM_RIGHT_TO_BOAT',
            'MOVE_WOLF_FROM_BOAT_TO_RIGHT',
            'MOVE_PLAYER_FROM_BOAT_TO_LEFT',
            'MOVE_PLAYER_FROM_BOAT_TO_RIGHT',
            'MOVE_PLAYER_FROM_LEFT_TO_BOAT',
            'MOVE_PLAYER_FROM_RIGHT_TO_BOAT',

        ]

    def move(self, state, what, where_from, where_to):
        state[where_from].remove(what)
        state[where_to].add(what)
        return None

    def get_next_states(self, starting_state):
        next_states = []
        for action in self.actions:
            next_state = copy.deepcopy(starting_state)

            if action == 'MOVE_PLAYER_FROM_LEFT_TO_BOAT':
                if '👨‍🌾' in next_state[0]:
                    self.move(next_state, '👨‍🌾' ,0 , 1)

            if action == 'MOVE_PLAYER_FROM_RIGHT_TO_BOAT':
                if '👨‍🌾' in next_state[2]:
                    self.move(next_state, '👨‍🌾', 2, 1)

            if action == 'MOVE_PLAYER_FROM_BOAT_TO_LEFT':
                if len(next_state[1]) == 1 and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1, 0)

            if action == 'MOVE_PLAYER_FROM_BOAT_TO_RIGHT':
                if len(next_state[1]) == 1 and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1, 2)

            if action == 'MOVE_GOAT_FROM_LEFT_TO_BOAT':
                if '🐐' in next_state[0] and '👨‍🌾' in next_state[0]:
                    self.move(next_state, '👨‍🌾', 0, 1)
                    self.move(next_state, '🐐', 0, 1)

            if action == 'MOVE_WOLF_FROM_LEFT_TO_BOAT':
                if '🐺' in next_state[0] and '👨‍🌾' in next_state[0]:
                    self.move(next_state, '👨‍🌾', 0, 1)
                    self.move(next_state, '🐺', 0, 1)

            if action == 'MOVE_CABBAGE_FROM_LEFT_TO_BOAT':
                if '🥦' in next_state[0] and '👨‍🌾' in next_state[0]:
                    self.move(next_state, '👨‍🌾', 0, 1)
                    self.move(next_state, '🥦', 0, 1)

            if action == 'MOVE_GOAT_FROM_BOAT_TO_LEFT':
                if '🐐' in next_state[1] and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1 ,0)
                    self.move(next_state, '🐐', 1, 0)

            if action == 'MOVE_WOLF_FROM_BOAT_TO_LEFT':
                if '🐺' in next_state[1] and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1, 0)
                    self.move(next_state, '🐺', 1, 0)

            if action == 'MOVE_CABBAGE_FROM_BOAT_TO_LEFT':
                if '🥦' in next_state[1] and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1, 0)
                    self.move(next_state, '🥦', 1, 0)

            if action == 'MOVE_GOAT_FROM_BOAT_TO_RIGHT':
                if '🐐' in next_state[1] and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1, 2)
                    self.move(next_state, '🐐', 1, 2)

            if action == 'MOVE_WOLF_FROM_BOAT_TO_RIGHT':
                if '🐺' in next_state[1] and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1, 2)
                    self.move(next_state, '🐺', 1, 2)

            if action == 'MOVE_CABBAGE_FROM_BOAT_TO_RIGHT':
                if '🥦' in next_state[1] and '👨‍🌾' in next_state[1]:
                    self.move(next_state, '👨‍🌾', 1, 2)
                    self.move(next_state, '🥦', 1, 2)

            if action == 'MOVE_GOAT_FROM_RIGHT_TO_BOAT':
                if '🐐' in next_state[2] and '👨‍🌾' in next_state[2]:
                    self.move(next_state, '👨‍🌾', 2, 1)
                    self.move(next_state, '🐐', 2, 1)

            if action == 'MOVE_WOLF_FROM_RIGHT_TO_BOAT':
                if '🐺' in next_state[2] and '👨‍🌾' in next_state[2]:
                    self.move(next_state, '👨‍🌾', 2, 1)
                    self.move(next_state, '🐺', 2, 1)

            if action == 'MOVE_CABBAGE_FROM_RIGHT_TO_BOAT':
                if '🥦' in next_state[2] and '👨‍🌾' in next_state[2]:
                    self.move(next_state, '👨‍🌾', 2, 1)
                    self.move(next_state, '🥦', 2, 1)

            next_states.append(next_state)

        return next_states

    def get_reward(self, state):
        # GOAT and WOLF cannot be left unsupervised together
        # GOAT and CABBAGE cannot be left unsupervised together
        if state[2] == self.goal_state[2]:
            return 100
        if {'🐐', '🐺'} <= state[0] and '👨‍🌾' not in state[0]:
            return -100
        if {'🐐', '🥦'} <= state[0] and '👨‍🌾' not in state[0]:
            return -100
        if {'🐐', '🐺'} <= state[2] and '👨‍🌾' not in state[2]:
            return -100
        if {'🐐', '🥦'} <= state[2] and '👨‍🌾' not in state[2]:
            return -100
        # goat cannot be in the boat alone
        if state[1] == {'🐐'}:
            return -100
        # wolf cannot be in the boat alone
        if state[1] == {'🐺'}:
            return -100
        if len(state[1]) > 2:
            # do not put more than 2 obj in the boat (player and max 1 obj)
            return -100
        return 0

    def train(self):

        convergence_count = 0
        q_s_a = defaultdict(int)
        q_s_a_prec = copy.deepcopy(q_s_a)
        episode = 1
        scores = []
        eps_list = []
        rewards = {}
        for episode in range(1,self.max_episodes):
            initial_state_for_this_episode = self.start_state
            score_per_episode = 0
            
            print("*** EPISODE {} ***".format(episode))
            while initial_state_for_this_episode != self.goal_state:

                next_states_for_action = self.get_next_states(initial_state_for_this_episode)
                for next_state_for_action in next_states_for_action:
                    k = RLKey(initial_state_for_this_episode, next_state_for_action)
                    rewards[k] = self.get_reward(next_state_for_action)
                chosen_next_state = rd.choice(next_states_for_action)

                if self.epsilon_greedy:
                    e = rd.uniform(0, 1)

                    if e > self.epsilon:
                        # action with max value from current state
                        # it's ok to randomly choose if every q is 0 because we would max on a full-0 list
                        s_a_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == initial_state_for_this_episode}
                        if len(s_a_list):
                            m = max(s_a_list, key=s_a_list.get)
                            chosen_next_state = m.k2

                q_s1_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == chosen_next_state}

                m_q_s1 = max(q_s1_list.values(), default=0)

                k_qsa = RLKey(initial_state_for_this_episode,chosen_next_state)
                q_s_a[k_qsa] = rewards[k_qsa] + (self.gamma * m_q_s1)
                score_per_episode += q_s_a[k_qsa]
                initial_state_for_this_episode = chosen_next_state

            if q_s_a == q_s_a_prec:
                if convergence_count > 10:
                    print('** CONVERGED **')
                    break
                else:
                    convergence_count += 1
            else:
                q_s_a_prec = copy.deepcopy(q_s_a)
                convergence_count = 0

            # epsilon update
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

            scores.append(score_per_episode)
            eps_list.append(self.epsilon)
           

        solution_steps = [self.start_state]
        next_state = self.start_state
        while next_state != self.goal_state:
            candidate_next_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == next_state}
            m = max(candidate_next_list, key=candidate_next_list.get)
            next_state = m.k2
            solution_steps.append(next_state)
        return solution_steps, scores, eps_list
