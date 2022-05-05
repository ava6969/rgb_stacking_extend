import gym


def main():

    env = gym.make('CartPole-v1')
    env.close()

if __name__ == '__main__':
    main()