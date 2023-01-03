import copy
import random as rd
import math
import numpy as np
from collections import defaultdict


class WolfGoatCabbageQLearning:

    def __init__(self, start_state, goal_state, gamma=0.8, max_episodes=50000, epsilon_greedy=True, min_epsilon=0.1, max_epsilon=1.0):
        self.start_state = start_state
        self.goal_state = goal_state
        self.gamma = gamma
        self.max_episodes = max_episodes

        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = 0.02
        self.epsilon = self.max_epsilon
        self.epsilon_greedy = epsilon_greedy

        self.actions = [
            'MOVE_GOAT_FROM_LEFT_TO_BOAT', 'MOVE_WOLF_FROM_LEFT_TO_BOAT', 'MOVE_CABBAGE_FROM_LEFT_TO_BOAT',
            'MOVE_GOAT_FROM_BOAT_TO_LEFT', 'MOVE_WOLF_FROM_BOAT_TO_LEFT', 'MOVE_CABBAGE_FROM_BOAT_TO_LEFT',
            'MOVE_GOAT_FROM_RIGHT_TO_BOAT', 'MOVE_WOLF_FROM_RIGHT_TO_BOAT', 'MOVE_CABBAGE_FROM_RIGHT_TO_BOAT',
            'MOVE_GOAT_FROM_BOAT_TO_RIGHT', 'MOVE_WOLF_FROM_BOAT_TO_RIGHT', 'MOVE_CABBAGE_FROM_BOAT_TO_RIGHT',
            'MOVE_PLAYER_FROM_BOAT_TO_LEFT',
            'MOVE_PLAYER_FROM_BOAT_TO_RIGHT',
            'MOVE_PLAYER_FROM_LEFT_TO_BOAT',
            'MOVE_PLAYER_FROM_RIGHT_TO_BOAT',

        ]

    def move_obj(self, banks, obj, towards):
        pass

    def get_next_states(self, starting_state):
        next_states = []
        for action in self.actions:
            next_state = copy.deepcopy(starting_state)
            is_legal = False
            if action == 'MOVE_PLAYER_FROM_LEFT_TO_BOAT':
                if 'P' in next_state[0]:
                    next_state[0].remove('P')
                    next_state[1].append('P')
                    is_legal = True
            if action == 'MOVE_PLAYER_FROM_RIGHT_TO_BOAT':
                if 'P' in next_state[2]:
                    next_state[2].remove('P')
                    next_state[1].append('P')
                    is_legal = True
            if action == 'MOVE_PLAYER_FROM_BOAT_TO_LEFT':
                if len(next_state[1]) == 1 and 'P' in next_state[1]:
                    next_state[1].remove('P')
                    next_state[0].append('P')
                    is_legal = True
            if action == 'MOVE_PLAYER_FROM_BOAT_TO_RIGHT':
                if len(next_state[1]) == 1 and 'P' in next_state[1]:
                    next_state[1].remove('P')
                    next_state[2].append('P')
                    is_legal = True

            if action == 'MOVE_GOAT_FROM_LEFT_TO_BOAT':
                if 'G' in next_state[0] and 'P' in next_state[0]:
                    next_state[0].remove('G')
                    next_state[1].append('G')
                    next_state[0].remove('P')
                    next_state[1].append('P')
                    is_legal = True

            if action == 'MOVE_WOLF_FROM_LEFT_TO_BOAT':
                if 'W' in next_state[0] and 'P' in next_state[0]:
                    next_state[0].remove('W')
                    next_state[1].append('W')
                    next_state[0].remove('P')
                    next_state[1].append('P')
                    is_legal = True

            if action == 'MOVE_CABBAGE_FROM_LEFT_TO_BOAT':
                if 'C' in next_state[0] and 'P' in next_state[0]:
                    next_state[0].remove('C')
                    next_state[1].append('C')
                    next_state[0].remove('P')
                    next_state[1].append('P')
                    is_legal = True

            if action == 'MOVE_GOAT_FROM_BOAT_TO_LEFT':
                if 'G' in next_state[1] and 'P' in next_state[1]:
                    next_state[1].remove('G')
                    next_state[0].append('G')
                    next_state[1].remove('P')
                    next_state[0].append('P')
                    is_legal = True

            if action == 'MOVE_WOLF_FROM_BOAT_TO_LEFT':
                if 'W' in next_state[1] and 'P' in next_state[1]:
                    next_state[1].remove('W')
                    next_state[0].append('W')
                    next_state[1].remove('P')
                    next_state[0].append('P')
                    is_legal = True

            if action == 'MOVE_CABBAGE_FROM_BOAT_TO_LEFT':
                if 'C' in next_state[1] and 'P' in next_state[1]:
                    next_state[1].remove('C')
                    next_state[0].append('C')
                    next_state[1].remove('P')
                    next_state[0].append('P')
                    is_legal = True

            if action == 'MOVE_GOAT_FROM_BOAT_TO_RIGHT':
                if 'G' in next_state[1] and 'P' in next_state[1]:
                    next_state[1].remove('G')
                    next_state[2].append('G')
                    next_state[1].remove('P')
                    next_state[2].append('P')
                    is_legal = True

            if action == 'MOVE_WOLF_FROM_BOAT_TO_RIGHT':
                if 'W' in next_state[1] and 'P' in next_state[1]:
                    next_state[1].remove('W')
                    next_state[2].append('W')
                    next_state[1].remove('P')
                    next_state[2].append('P')
                    is_legal = True

            if action == 'MOVE_CABBAGE_FROM_BOAT_TO_RIGHT':
                if 'C' in next_state[1] and 'P' in next_state[1]:
                    next_state[1].remove('C')
                    next_state[2].append('C')
                    next_state[1].remove('P')
                    next_state[2].append('P')
                    is_legal = True

            if action == 'MOVE_GOAT_FROM_RIGHT_TO_BOAT':
                if 'G' in next_state[2] and 'P' in next_state[2]:
                    next_state[2].remove('G')
                    next_state[1].append('G')
                    next_state[2].remove('P')
                    next_state[1].append('P')
                    is_legal = True

            if action == 'MOVE_WOLF_FROM_RIGHT_TO_BOAT':
                if 'W' in next_state[2] and 'P' in next_state[2]:
                    next_state[2].remove('W')
                    next_state[1].append('W')
                    next_state[2].remove('P')
                    next_state[1].append('P')
                    is_legal = True

            if action == 'MOVE_CABBAGE_FROM_RIGHT_TO_BOAT':
                if 'C' in next_state[2] and 'P' in next_state[2]:
                    next_state[2].remove('C')
                    next_state[1].append('C')
                    next_state[2].remove('P')
                    next_state[1].append('P')
                    is_legal = True

            if is_legal:
                next_state[0] = sorted(next_state[0])
                next_state[1] = sorted(next_state[1])
                next_state[2] = sorted(next_state[2])
                next_states.append(next_state)
        return next_states

    def get_reward(self, state):
        # GOAT and WOLF cannot be left unsupervised together
        # GOAT and CABBAGE cannot be left unsupervised together
        if sorted(state[2]) == sorted(self.goal_state[2]):
            return 100
        if 'G' in state[0] and 'W' in state[0] and not 'P' in state[0]:
            return -100
        if 'G' in state[0] and 'C' in state[0] and not 'P' in state[0]:
            return -100
        if 'G' in state[2] and 'W' in state[2] and not 'P' in state[2]:
            return -100
        if 'G' in state[2] and 'C' in state[2] and not 'P' in state[2]:
            return -100
        # goat cannot be in the boat alone
        if 'G' in state[1] and len(state[1]) == 1:
            return -100
        # wolf cannot be in the boat alone
        if 'W' in state[1] and len(state[1]) == 1:
            return -100
        if len(state[1]) > 2:
            # do not put more than 2 obj in the boat (player and max 1 obj)
            return -100
        return 0

    def train(self, verbose=False):

        GAMMA = self.gamma
        GOAL_STATE = self.goal_state
        MAX_EPISODES = self.max_episodes

        convergence_count = 0
        q_s_a = defaultdict(lambda: 0)
        q_s_a_prec = copy.deepcopy(q_s_a)
        episode = 1
        # rd.seed(42)
        scores = []
        eps_list = []
        rewards = {}
        while episode <= MAX_EPISODES:
            initial_state_for_this_episode = self.start_state
            score_per_episode = 0
            print('*** EPISODE '+str(episode)+' ***')
            while initial_state_for_this_episode != GOAL_STATE:

                next_states_for_action = self.get_next_states(
                    initial_state_for_this_episode)
                for next_state_for_action in next_states_for_action:
                    rewards[str(initial_state_for_this_episode)+"|"+str(next_state_for_action)
                            ] = self.get_reward(next_state_for_action)
                chosen_next_state = str(rd.choice(
                    next_states_for_action))
                # print('next_states_for_action')
                # print(next_states_for_action)

                if self.epsilon_greedy:
                    e = rd.uniform(0, 1)

                    if e > self.epsilon:
                        # action with max value from current state
                        # it's ok to randomly choose if every q is 0 because we would max on a full-0 list
                        s_a_list = {x: q_s_a[x] for x in q_s_a.keys() if x.startswith(
                            str(initial_state_for_this_episode)+"|")}
                        if len(s_a_list) > 0:
                            m = max(
                                s_a_list, key=s_a_list.get)
                            chosen_next_state = m.split('|')[1]

                q_s1_list = {x: q_s_a[x] for x in q_s_a.keys() if x.startswith(
                    chosen_next_state+"|")}

                if len(q_s1_list) > 0:
                    m_q_s1 = max(q_s1_list.values())
                else:
                    m_q_s1 = 0

                q_s_a[str(initial_state_for_this_episode) + "|" +
                      chosen_next_state] = rewards[str(initial_state_for_this_episode) + "|" + chosen_next_state] + (GAMMA * m_q_s1)
                score_per_episode += q_s_a[str(initial_state_for_this_episode) + "|" +
                                           chosen_next_state]
                initial_state_for_this_episode = eval(chosen_next_state)

            if q_s_a == q_s_a_prec:
                if convergence_count > int(10):
                    print('** CONVERGED **')
                    break
                else:
                    convergence_count += 1
            else:
                q_s_a_prec = copy.deepcopy(q_s_a)
                convergence_count = 0

            # epsilon update
            self.epsilon = self.min_epsilon + \
                (self.max_epsilon - self.min_epsilon) * \
                np.exp(-self.decay_rate * episode)

            scores.append(score_per_episode)
            eps_list.append(self.epsilon)
            episode += 1

        c = 0

        print(q_s_a)

        print('*** SCORES ***')
        for score, e in zip(scores, eps_list):
            print(str(score).replace(".", ",") +
                  ";"+str(e*100).replace(".", ","))
        print(self.start_state)
        next_state = str(self.start_state)
        while next_state != str(self.goal_state) and c < 100:
            candidate_next_list = {x: q_s_a[x] for x in q_s_a.keys() if x.startswith(
                next_state+"|")}
            m = max(
                candidate_next_list, key=candidate_next_list.get)
            next_state = m.split("|")[1]
            print(next_state)
            c += 1
