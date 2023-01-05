
from wolf_goat_cabbage_qlearning import WolfGoatCabbageQLearning


def main():
    wcg_arena = WolfGoatCabbageQLearning(
        start_state=[['🥦', '🐐', '👨‍🌾', '🐺'], [], []],
        goal_state=[[], [], ['👨‍🌾', '🐐', '🥦', '🐺']],
        gamma=0.8,
        max_episodes=100,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = wcg_arena.train()

    print('*** SOLUTION ***')
    for step in solution_steps:
        print(step)

    print('*** SCORES ***')
    for score, e in zip(scores, eps_list):
        print("{score:.2f};{epsilon:.2f}".format(score=score, epsilon=e))


if __name__ == "__main__":
    main()
