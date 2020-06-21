"""
    This file is copied/apdated from 2020 NCTU DLP assignment
"""
__author__ = 'chengscott and elephanting'
import argparse
from tqdm import tqdm
import random
import gym
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter

from model import DQN

def train(args, env, agent, writer=None):
    action_space = env.action_space
    total_steps = 0
    start_epoch = 1
    replay_initial = args.warmup
    epsilon_by_steps = lambda steps, replay_start_time: 1 - 0.9 * min(replay_start_time, args.epsilon)/args.epsilon

    if args.resume:
        if args.model is None:
            raise NameError('Please input model path')
        agent.load(args.model)
        total_steps = agent.total_steps
        start_epoch = agent.epoch + 1

    for epoch in range(start_epoch, args.epochs+1):
        print('Start Training')
        total_reward = 0
        state = env.reset()
        rewards = []

        for t in tqdm(range(1, args.steps+1)):
            # select action
            epsilon = epsilon_by_steps(total_steps, max([total_steps-replay_initial, 0]))
            action = agent.select_action(state, epsilon, action_space)
            # execute action
            next_state, reward, done, _ = env.step(action)
            
            # clip reward
            reward = np.clip(reward, -1, 1)
            
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                #writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                state = env.reset()
                rewards.append(total_reward)
                total_reward = 0

            if t % 100000 == 0:
                print('epoch: {}, total steps: {}, average reward: {:.2f}, epsilon: {:.2f}'.format(epoch, total_steps, np.mean(rewards[-100:]), epsilon))
                rewards = []            
        
        if args.checkpoint:
            agent.epoch = epoch
            agent.total_steps = total_steps
            agent.save('model/weight.tar')
        
        # every test need about 30 mins
        test(args, env, agent, epoch, writer)
    env.close()


def test(args, env, agent, epoch, writer):
    print('Start Testing, every test needs about 30 mins')
    action_space = env.action_space
    rewards = []
    total_reward = 0
    state = env.reset()

    mean_q_val = agent.get_mean_q_val()

    for step in tqdm(range(args.test_steps)):
        if args.render:
            env.render()
        action = agent.select_action(state, args.test_epsilon, action_space)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    
        if done:
            #writer.add_scalar('Test/Episode Reward', total_reward, step)
            rewards.append(total_reward)
            total_reward = 0
            state = env.reset()
    
    # last element of rewards is mean Q value
    rewards.append(mean_q_val)

    print('Average Testing Reward:', np.mean(rewards))
    print('Average Q value: {:.2f}'.format(mean_q_val))
    np.save('result/test_{}.npy'.format(epoch), rewards)
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--checkpoint', action='store_true', help='save model every epoch, ER buffer will only be saved as one file')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--steps', default=1000000, type=int, help='steps per epoch')
    parser.add_argument('--capacity', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--epsilon', default=1000000, type=int, help='decay steps from eps_max to eps_min')
    parser.add_argument('--eps_min', default=.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_steps', type=int, default=500000)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_epsilon', default=0.05, type=float)
    # average
    parser.add_argument('-k', '--k', type=int, default=1, help='number of average target network, if k = 1, perform vanilla DQN')
    # resume training
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('-m', '--model', default=None, help='model path')
    # DDQN
    parser.add_argument('--ddqn', action='store_true', help='perform Double-DQN')

    args = parser.parse_args()

    ## main ##
    env = gym.make('BreakoutNoFrameskip-v4')
    
    # frame stack and preprocessing
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4)
    env = FrameStack(env, 4)

    agent = DQN(args, env)
    #writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent)
        agent.save(args.model)


if __name__ == '__main__':
    main()
