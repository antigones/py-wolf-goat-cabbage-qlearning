# py-wolf-goat-cabbage-qlearning
"Cabbage, goat, wolf" river crossing puzzle, solved with q-learning πΊπ₯¦πβ΅

This class uses determistic MDP formula to compute Q (and the optimal policy).

Example usage:

    wcg_arena = WolfGoatCabbageQLearning(
        start_state=[['π₯¦', 'π', 'β΅', 'πΊ'], [], []],
        goal_state=[[], [], ['β΅', 'π', 'π₯¦', 'πΊ']],
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = wcg_arena.train()

Example solution:

    *** SOLUTION ***
    ({'π', 'β΅', 'πΊ', 'π₯¦'}, set(), set())
    ({'πΊ', 'π₯¦'}, {'π', 'β΅'}, set())
    ({'πΊ', 'π₯¦'}, set(), {'π', 'β΅'})
    ({'πΊ', 'π₯¦'}, {'β΅'}, {'π'})
    ({'β΅', 'πΊ', 'π₯¦'}, set(), {'π'})
    ({'πΊ'}, {'β΅', 'π₯¦'}, {'π'})
    ({'πΊ'}, set(), {'π', 'β΅', 'π₯¦'})
    ({'πΊ'}, {'π', 'β΅'}, {'π₯¦'})
    ({'π', 'β΅', 'πΊ'}, set(), {'π₯¦'})
    ({'π'}, {'β΅', 'πΊ'}, {'π₯¦'})
    ({'π'}, set(), {'β΅', 'πΊ', 'π₯¦'})
    ({'π'}, {'β΅'}, {'πΊ', 'π₯¦'})
    ({'π', 'β΅'}, set(), {'πΊ', 'π₯¦'})
    (set(), {'π', 'β΅'}, {'πΊ', 'π₯¦'})
    (set(), set(), {'π', 'β΅', 'πΊ', 'π₯¦'})

Parameters:

- start_state=`[['π₯¦', 'π', 'β΅', 'πΊ'], [], []]`
- goal_state=`[[], [], ['β΅', 'π', 'π₯¦', 'πΊ']]`
- gamma=0.8
- max_episodes=50000
- epsilon_greedy=True (specifies if you want to apply an e-greedy policy to compute Q)
    - min_epsilon=0.1
    - max_epsilon=1.0

Example graph for cumulative discounted reward and epsilon value per episode:

![Reward/epsilon per episode](/images/reward_epsilon.png)

