import torch
import gym
import argparse

from model import DQN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path of input test weight')
    parser.add_argument('--rounds', type=int, default=3, help='play x rounds')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_epsilon', default=0, type=float)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('BreakoutNoFrameskip-v4')
    
    # frame stack and preprocessing
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4)
    env = FrameStack(env, 4)

    model = DQN(env.observation_space.shape, env.action_space.n).to(device).eval()
    model.load(args.path, test=True)

    # play three rounds
    for i in range(args.rounds):
        done = False
        total_reward = 0
        state = env.reset()
        
        while not done:
            if args.render:
                env.render()
            action = model.select_action(state, args.test_epsilon, action_space)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                print(total_reward)
                break