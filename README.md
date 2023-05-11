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
    solution_steps, actions, scores, eps_list = wcg_arena.train()

Example solution:

    *** SOLUTION ***
    ({'🐐', '⛵', '🥦', '🐺'}, set(), set())
    ({'🐺', '🥦'}, {'⛵', '🐐'}, set())
    ({'🐺', '🥦'}, set(), {'⛵', '🐐'})
    ({'🐺', '🥦'}, {'⛵'}, {'🐐'})
    ({'🐺', '⛵', '🥦'}, set(), {'🐐'})
    ({'🥦'}, {'🐺', '⛵'}, {'🐐'})
    ({'🥦'}, set(), {'🐺', '⛵', '🐐'})
    ({'🥦'}, {'⛵', '🐐'}, {'🐺'})
    ({'🐐', '⛵', '🥦'}, set(), {'🐺'})
    ({'🐐'}, {'⛵', '🥦'}, {'🐺'})
    ({'🐐'}, set(), {'🐺', '⛵', '🥦'})
    ({'🐐'}, {'⛵'}, {'🐺', '🥦'})
    ({'⛵', '🐐'}, set(), {'🐺', '🥦'})
    (set(), {'⛵', '🐐'}, {'🐺', '🥦'})
    (set(), set(), {'🐺', '⛵', '🥦', '🐐'})

    *** ACTIONS ***
    MOVE_GOAT_AND_PLAYER_FROM_LEFT_TO_BOAT
    MOVE_GOAT_AND_PLAYER_FROM_BOAT_TO_RIGHT
    MOVE_PLAYER_FROM_RIGHT_TO_BOAT
    MOVE_PLAYER_FROM_BOAT_TO_LEFT
    MOVE_WOLF_AND_PLAYER_FROM_LEFT_TO_BOAT
    MOVE_WOLF_AND_PLAYER_FROM_BOAT_TO_RIGHT
    MOVE_GOAT_AND_PLAYER_FROM_RIGHT_TO_BOAT
    MOVE_GOAT_AND_PLAYER_FROM_BOAT_TO_LEFT
    MOVE_CABBAGE_AND_PLAYER_FROM_LEFT_TO_BOAT
    MOVE_CABBAGE_AND_PLAYER_FROM_BOAT_TO_RIGHT
    MOVE_PLAYER_FROM_RIGHT_TO_BOAT
    MOVE_PLAYER_FROM_BOAT_TO_LEFT
    MOVE_GOAT_AND_PLAYER_FROM_LEFT_TO_BOAT
    MOVE_GOAT_AND_PLAYER_FROM_BOAT_TO_RIGHT

Parameters:

- start_state=`[['🥦', '🐐', '⛵', '🐺'], [], []]`
- goal_state=`[[], [], ['⛵', '🐐', '🥦', '🐺']]`
- gamma=0.8
- max_episodes=50000
- epsilon_greedy=True (specifies if you want to apply an e-greedy policy to compute Q)
    - min_epsilon=0.1
    - max_epsilon=1.0

Example graph for total reward and epsilon value per episode:

![Reward/epsilon per episode](/images/reward_epsilon.png)

