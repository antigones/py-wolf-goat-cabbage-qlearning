# py-wolf-goat-cabbage-qlearning
"Cabbage, goat, wolf" river crossing puzzle, solved with q-learning 🐺🥦🐐⛵

This class uses determistic MDP formula to compute Q (and the optimal policy).

Example usage:

    wcg_arena = WolfGoatCabbageQLearning(
        start_state=[['🥦', '🐐', '⛵', '🐺'], [], []],
        goal_state=[[], [], ['⛵', '🐐', '🥦', '🐺']],
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = wcg_arena.train()

Example solution:

    *** SOLUTION ***
    ({'⛵', '🐐', '🐺', '🥦'}, set(), set())
    ({'🐺', '🥦'}, {'⛵', '🐐'}, set())
    ({'🐺', '🥦'}, set(), {'⛵', '🐐'})
    ({'🐺', '🥦'}, {'⛵'}, {'🐐'})
    ({'⛵', '🐺', '🥦'}, set(), {'🐐'})
    ({'🐺'}, {'⛵', '🥦'}, {'🐐'})
    ({'🐺'}, set(), {'⛵', '🐐', '🥦'})
    ({'🐺'}, {'⛵', '🐐'}, {'🥦'})
    ({'⛵', '🐺', '🐐'}, set(), {'🥦'})
    ({'🐐'}, {'⛵', '🐺'}, {'🥦'})
    ({'🐐'}, set(), {'⛵', '🐺', '🥦'})
    ({'🐐'}, {'⛵'}, {'🐺', '🥦'})
    ({'⛵', '🐐'}, set(), {'🐺', '🥦'})
    (set(), {'⛵', '🐐'}, {'🐺', '🥦'})
    (set(), set(), {'⛵', '🐺', '🐐', '🥦'})

Parameters:

- start_state=`[['🥦', '🐐', '⛵', '🐺'], [], []]`
- goal_state=`[[], [], ['⛵', '🐐', '🥦', '🐺']]`
- gamma=0.8
- max_episodes=50000
- epsilon_greedy=True (specifies if you want to apply an e-greedy policy to compute Q)
    - min_epsilon=0.1
    - max_epsilon=1.0

Example graph for cumulative discounted reward and epsilon value per episode:

![Reward/epsilon per episode](/images/reward_epsilon.png)

