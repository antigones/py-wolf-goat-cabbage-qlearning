# py-wolf-goat-cabbage-qlearning
"Cabbage, goat, wolf" river crossing puzzle, solved with q-learning 🐺🥦🐐⛵

This class uses determistic MDP formula to compute Q (and the optimal policy).

Example usage:

    wcg_arena = WolfGoatCabbageQLearning(
        start_state=[['C', 'G', 'P', 'W'], [], []],
        goal_state=[[], [], ['C', 'G', 'P', 'W']],
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = wcg_arena.train()

Example solution:

    *** SOLUTION ***
    [['C', 'G', 'P', 'W'], [], []]
    [['C', 'W'], ['G', 'P'], []]
    [['C', 'W'], [], ['G', 'P']]
    [['C', 'W'], ['P'], ['G']]
    [['C', 'P', 'W'], [], ['G']]
    [['C'], ['P', 'W'], ['G']]
    [['C'], [], ['G', 'P', 'W']]
    [['C'], ['G', 'P'], ['W']]
    [['C', 'G', 'P'], [], ['W']]
    [['G'], ['C', 'P'], ['W']]
    [['G'], [], ['C', 'P', 'W']]
    [['G'], ['P'], ['C', 'W']]
    [['G', 'P'], [], ['C', 'W']]
    [[], ['G', 'P'], ['C', 'W']]
    [[], [], ['C', 'G', 'P', 'W']]

Parameters:

- start_state=`[['C', 'G', 'P', 'W'], [], []]`
- goal_state=`[[], [], ['C', 'G', 'P', 'W']]`
- gamma=0.8
- max_episodes=50000
- epsilon_greedy=True (specifies if you want to apply an e-greedy policy to compute Q)
    - min_epsilon=0.1
    - max_epsilon=1.0

Example graph for cumulative discounted reward and epsilon value per episode:

![Reward/epsilon per episode](/images/reward_epsilon.png)

