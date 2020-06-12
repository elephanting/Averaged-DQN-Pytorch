import torch
import gym
import argparse

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from DQN import DQN, device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path of input test weight')
    parser.add_argument('--rounds', type=int, default=3, help='play x rounds')
    args = parser.parse_args()

    env_id = "BreakoutNoFrameskip-v4"
    env    = make_atari(env_id)
    env    = wrap_deepmind(env)
    env    = wrap_pytorch(env)

    test = torch.load(args.path)
    model = DQN(env.observation_space.shape, env.action_space.n).to(device).eval()
    model.load_state_dict(test['model'])

    # play three rounds
    for i in range(args.rounds):
        done = False
        state = env.reset()
        
        while not done:
            env.render()
            action = model.act(state, 0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break