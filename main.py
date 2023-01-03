from wolf_goat_cabbage_qlearning import WolfGoatCabbageQLearning


def main():

    wcg_arena = WolfGoatCabbageQLearning(
        start_state=[['C', 'G', 'P', 'W'], [], []],
        goal_state=[[], [], ['C', 'G', 'P', 'W']],
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    wcg_arena.train(verbose=True)
    # print(state_space)


if __name__ == "__main__":
    main()
