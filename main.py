from wolf_goat_cabbage_qlearning import WolfGoatCabbageQLearning


def main():
    wcg_arena = WolfGoatCabbageQLearning(
        start_state=[['C', 'G', 'P', 'W'], [], []],
        goal_state=[[], [], ['C', 'G', 'P', 'W']],
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = wcg_arena.train()

    print('*** SOLUTION ***')
    for step in solution_steps:
        print(step)

    score_text = "{score:.2f};{epsilon:.2f}"
    print('*** SCORES ***')
    for score, e in zip(scores, eps_list):
        print(score_text.format(score=score, epsilon=e))


if __name__ == "__main__":
    main()
